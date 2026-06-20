//! Local LLM assistant engine for Corylus.
//!
//! Architecture mirrors the other Corylus vendor crates (rust-pos, rust-lemma, etc.):
//!   - All assistant logic lives here; the app contains only thin Tauri command glue.
//!   - Download-on-demand: Ollama + a Qwen3 model are fetched at runtime, never bundled.
//!   - Backend abstraction: `AssistantBackend` trait; Phase 1 = `OllamaBackend`.
//!   - State machine: not_installed → needs_install/server_down → downloading → ready | error.
//!   - RAG grounding: sqlite-vec index over manuscript + story-bible.

mod error;
mod state;
mod backend;
mod ollama;
mod provision;
pub mod rag;
pub mod index;

#[cfg(test)]
mod tests;

pub use error::AssistantError;
pub use state::AssistantState;
pub use backend::{AssistantBackend, ChatMessage, ChatToken, RamTier, RetrievedChunk, build_system_prompt};
pub use ollama::OllamaBackend;
pub use index::{VectorIndex, RetrievedRow};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub type Result<T> = std::result::Result<T, AssistantError>;

/// Configuration for the assistant engine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Directory under the app data dir where the engine stores its assets.
    pub data_dir: PathBuf,
    /// Ollama model tag to use (e.g. "qwen3:8b").
    pub model_id: String,
    /// Embedding model for RAG (e.g. "nomic-embed-text").
    pub embedding_model: String,
    /// Ollama API endpoint (default: "http://127.0.0.1:11434").
    pub ollama_endpoint: String,
    /// Context window size passed to Ollama (num_ctx). Default 4096.
    pub context_window: u32,
}

impl EngineConfig {
    pub fn default_for_ram(data_dir: PathBuf, ram_bytes: Option<u64>) -> Self {
        let tier = RamTier::from_bytes(ram_bytes);
        Self {
            data_dir,
            model_id: tier.default_model().into(),
            embedding_model: "nomic-embed-text".into(),
            ollama_endpoint: "http://127.0.0.1:11434".into(),
            context_window: 4096,
        }
    }
}

/// Sync-only shared state (config + current state machine state).
pub(crate) struct Inner {
    pub config: EngineConfig,
    pub state: AssistantState,
}

/// The assistant engine. Cheap to clone (Arc-backed).
///
/// The backend uses a tokio Mutex so it can be locked across `.await` without
/// violating the `Send` bound on `tauri::async_runtime::spawn`.
#[derive(Clone)]
pub struct AssistantEngine {
    inner: Arc<Mutex<Inner>>,
    backend: Arc<tokio::sync::Mutex<OllamaBackend>>,
}

impl AssistantEngine {
    pub fn new(config: EngineConfig) -> Self {
        let backend = OllamaBackend::new(config.clone());
        Self {
            inner: Arc::new(Mutex::new(Inner {
                state: AssistantState::NotInstalled,
                config,
            })),
            backend: Arc::new(tokio::sync::Mutex::new(backend)),
        }
    }

    /// Update data dir. Called from app setup once the resolved app-data dir is known.
    pub fn set_data_dir(&self, data_dir: PathBuf) {
        let mut g = self.inner.lock().unwrap();
        g.config.data_dir = data_dir.clone();
        // Probe the backend synchronously; try_lock is safe here since no async
        // task holds the backend lock at startup.
        if let Ok(mut b) = self.backend.try_lock() {
            b.config.data_dir = data_dir.clone();
            let s = b.probe();
            g.state = s;
            // Restore engine_managed_install from the persisted version sentinel so
            // that subsequent `ollama serve` invocations keep OLLAMA_MODELS set correctly.
            if let Ok(ver) = state::read_version_file(&g.config) {
                b.engine_managed_install = ver.ollama_managed;
            }
        }
        if state::is_ready(&g.config) {
            g.state = AssistantState::Ready;
        }
    }

    pub fn state(&self) -> AssistantState {
        self.inner.lock().unwrap().state.clone()
    }

    pub fn data_dir(&self) -> PathBuf {
        self.inner.lock().unwrap().config.data_dir.clone()
    }

    pub fn config(&self) -> EngineConfig {
        self.inner.lock().unwrap().config.clone()
    }

    /// Begin provisioning (install Ollama + pull model). Consent must be given before calling.
    pub async fn provision(
        &self,
        on_progress: impl Fn(AssistantState) + Send + Sync + 'static,
    ) -> Result<()> {
        provision::run(self.inner.clone(), self.backend.clone(), on_progress).await
    }

    /// Detect available system RAM (best-effort).
    pub fn detect_ram_bytes(&self) -> Option<u64> {
        self.backend.try_lock().ok().as_mut().map(|b| b.detect_ram_bytes()).flatten()
    }

    /// Return the RAM-aware default model tag and its approximate size/RAM.
    pub fn model_info(&self) -> ModelInfo {
        let ram = self.detect_ram_bytes();
        let tier = RamTier::from_bytes(ram);
        ModelInfo {
            model_id: tier.default_model().into(),
            download_size_bytes: tier.default_model_size_bytes(),
            ram_tier: format!("{:?}", tier).to_lowercase(),
        }
    }

    /// Uninstall: remove the version sentinel (model files stay in Ollama's store).
    pub fn uninstall(&self) -> Result<()> {
        let g = self.inner.lock().unwrap();
        let ver = state::version_path(&g.config);
        if ver.exists() {
            std::fs::remove_file(&ver)?;
        }
        drop(g);
        self.inner.lock().unwrap().state = AssistantState::NotInstalled;
        Ok(())
    }

    /// Index a set of project documents for RAG retrieval.
    ///
    /// `documents` is a list of `(relative_path, plain_text_content)` pairs.
    /// Pass `full_reindex = true` to wipe the existing index first.
    /// Returns `Err(EmbeddingModelMismatch)` if the configured embedding model
    /// changed since the index was last built — caller must set `full_reindex = true`.
    pub async fn index_project(
        &self,
        documents: Vec<(String, String)>,
        full_reindex: bool,
        on_progress: impl Fn(usize, usize) + Send + 'static,
    ) -> Result<()> {
        let config = self.config();
        let mut idx = index::VectorIndex::open(&config)?;

        if !full_reindex {
            idx.check_model_compatibility(&config)?;
        }

        // The embed closure captures a clone of the backend Arc so it can be called
        // from within the async indexing loop without holding any outer lock.
        let backend = self.backend.clone();

        struct BackendEmbedder(Arc<tokio::sync::Mutex<OllamaBackend>>);
        impl rag::AsyncEmbedFn for BackendEmbedder {
            async fn embed(&self, text: &str) -> Result<Vec<f32>> {
                let b = self.0.lock().await;
                backend::AssistantBackend::embed(&*b, text).await
            }
        }

        let embedder = BackendEmbedder(backend);

        if full_reindex {
            idx.full_reindex(documents, &config, &embedder, on_progress).await
        } else {
            idx.incremental_update(documents, &config, &embedder, on_progress).await
        }
    }

    /// Return the number of indexed chunks (0 = index empty or not yet built).
    pub fn chunk_count(&self) -> Result<usize> {
        let config = self.config();
        let idx = index::VectorIndex::open(&config)?;
        Ok(idx.chunk_count()?)
    }

    /// Send a chat message grounded in the RAG index and return the full response text.
    /// Retrieves the top-5 relevant chunks, builds a system prompt, and calls the backend.
    pub async fn chat(&self, user_message: &str) -> Result<String> {
        // Retrieve relevant context and convert to RetrievedChunk for the system prompt builder
        let rows = self.retrieve(user_message, 5).await.unwrap_or_default();
        let chunks: Vec<backend::RetrievedChunk> = rows.into_iter().map(|r| backend::RetrievedChunk {
            text: r.text,
            source: r.source,
            score: (1.0 - r.distance as f32).max(0.0),
        }).collect();
        let system_prompt = backend::build_system_prompt(&chunks);

        let messages = vec![
            ChatMessage { role: "system".into(), content: system_prompt },
            ChatMessage { role: "user".into(),   content: user_message.into() },
        ];

        // Use a one-shot channel: send all tokens, collect into a string.
        let (token_tx, token_rx) = std::sync::mpsc::channel::<ChatToken>();
        let (_cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();

        let backend = self.backend.clone();
        let messages_clone = messages.clone();
        let handle = tokio::spawn(async move {
            let mut b = backend.lock().await;
            backend::AssistantBackend::chat(&mut *b, messages_clone, token_tx, cancel_rx).await
        });

        // Collect tokens synchronously from the receiver
        let mut full_text = String::new();
        for token in token_rx {
            full_text.push_str(&token.text);
            if token.done { break; }
        }
        handle.await.map_err(|e| AssistantError::Internal(e.to_string()))??;
        Ok(full_text)
    }

    /// Retrieve the top-K most relevant chunks for `query`, using the RAG index.
    /// Returns an empty vec when the index has no content or the engine isn't ready.
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedRow>> {
        let config = self.config();
        let idx = index::VectorIndex::open(&config)?;
        if idx.chunk_count()? == 0 {
            return Ok(vec![]);
        }
        // Embed the query with the same model used to build the index
        let query_vec = self.backend.lock().await.embed(query).await?;
        idx.top_k(&query_vec, top_k)
    }
}

/// Info about the recommended model for display in the UI before download.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub download_size_bytes: u64,
    pub ram_tier: String,
}

// Shim so RamTier serializes as a string
impl serde::Serialize for RamTier {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            RamTier::Low => "low",
            RamTier::Medium => "medium",
            RamTier::High => "high",
        })
    }
}

impl<'de> serde::Deserialize<'de> for RamTier {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "medium" => RamTier::Medium,
            "high" => RamTier::High,
            _ => RamTier::Low,
        })
    }
}
