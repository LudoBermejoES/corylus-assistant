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
pub mod actions;
pub mod catalog;
pub mod rag;
pub mod index;
pub mod fallback;
pub mod router;

#[cfg(test)]
mod tests;

pub use error::AssistantError;
pub use state::AssistantState;
pub use backend::{AssistantBackend, ChatMessage, ChatToken, RamTier, RetrievedChunk, build_system_prompt};
pub use ollama::OllamaBackend;
pub use index::{VectorIndex, RetrievedRow};
pub use catalog::{ModelEntry, CATALOG};
pub use fallback::{FallbackEngine, FsmState, FsmFlow};
pub use router::{Router, RouterResponse, is_llm_ready};
pub use actions::{ProposedAction, ACTION_CATALOG, find_action};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

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
    cancel_pull: Arc<AtomicBool>,
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
            cancel_pull: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal any in-progress model pull to stop after the current chunk.
    pub fn cancel_pull(&self) {
        self.cancel_pull.store(true, Ordering::Relaxed);
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

    /// Override the chat model ID. Takes effect on the next chat/provision call.
    pub fn set_model_id(&self, model_id: String) {
        let mut g = self.inner.lock().unwrap();
        g.config.model_id = model_id.clone();
        if let Ok(mut b) = self.backend.try_lock() {
            b.config.model_id = model_id;
        }
        // If try_lock failed (backend busy), the backend config is synced from inner.config
        // at the start of every chat() and provision() call.
    }

    /// Override the embedding model ID. Takes effect on the next index/embed call.
    pub fn set_embedding_model(&self, embedding_model: String) {
        let mut g = self.inner.lock().unwrap();
        g.config.embedding_model = embedding_model.clone();
        if let Ok(mut b) = self.backend.try_lock() {
            b.config.embedding_model = embedding_model;
        }
        // If try_lock failed, the backend config is synced from inner.config at call time.
    }

    /// Begin provisioning (install Ollama + pull model). Consent must be given before calling.
    pub async fn provision(
        &self,
        on_progress: impl Fn(AssistantState) + Send + Sync + 'static,
    ) -> Result<()> {
        self.cancel_pull.store(false, Ordering::Relaxed);
        provision::run(self.inner.clone(), self.backend.clone(), self.cancel_pull.clone(), on_progress).await
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

    /// Return names of all locally downloaded Ollama models.
    pub async fn list_local_models(&self) -> Result<Vec<String>> {
        let b = self.backend.lock().await;
        b.list_local_models().await
    }

    /// Delete a model from Ollama's local store.
    pub async fn delete_model(&self, model_id: String) -> Result<()> {
        let b = self.backend.lock().await;
        b.delete_model(&model_id).await
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

        // Ensure the Ollama server is running before starting the embed loop.
        // Without this, all embed calls fail with connection refused if the server
        // hasn't been started yet (e.g. indexing triggered before any chat).
        tracing::info!("[assistant] index_project: ensuring server running before embedding");
        {
            let mut b = self.backend.lock().await;
            b.ensure_server_running().await?;
            // Pull the embedding model if not yet present (e.g. existing installs that
            // predate the auto-pull logic, or first index after a fresh chat-model install).
            b.ensure_embedding_model().await?;
        }
        tracing::info!("[assistant] index_project: server ready, starting embed loop");

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
        tracing::info!("[assistant] chat() called with message: {:?}", &user_message[..user_message.len().min(80)]);
        // Retrieve relevant context — falls back to empty if server not yet up or index empty.
        // The actual server startup happens inside the backend chat() call below.
        let rows = self.retrieve(user_message, 5).await.unwrap_or_default();
        tracing::info!("[assistant] chat() retrieved {} context chunks", rows.len());
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

        let (_cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
        // Use an async channel so receiving tokens doesn't block the Tokio worker thread.
        let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<ChatToken>();

        // Sync model_id from inner.config into the backend at call time.
        // This ensures set_model_id() always takes effect even if try_lock failed earlier.
        let model_id = self.inner.lock().unwrap().config.model_id.clone();
        let backend = self.backend.clone();
        let messages_clone = messages.clone();
        let handle = tokio::spawn(async move {
            let mut b = backend.lock().await;
            b.config.model_id = model_id;
            tracing::info!("[assistant] chat() backend lock acquired, calling chat");
            backend::AssistantBackend::chat(&mut *b, messages_clone, token_tx, cancel_rx).await
        });

        // Collect tokens asynchronously — does not block the Tokio worker thread.
        let mut full_text = String::new();
        while let Some(token) = token_rx.recv().await {
            full_text.push_str(&token.text);
            if token.done { break; }
        }
        handle.await.map_err(|e| AssistantError::Internal(e.to_string()))??;
        tracing::info!("[assistant] chat() completed, response length={}", full_text.len());
        Ok(full_text)
    }

    /// Retrieve the top-K most relevant chunks for `query`, using the RAG index.
    /// Returns an empty vec when the index has no content or the engine isn't ready.
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedRow>> {
        let config = self.config();
        let idx = index::VectorIndex::open(&config)?;
        if idx.chunk_count()? == 0 {
            tracing::info!("[assistant] retrieve: index empty, skipping");
            return Ok(vec![]);
        }
        // Check server is alive before embedding — avoids hanging on a dead server.
        tracing::info!("[assistant] retrieve: acquiring backend lock for is_server_alive");
        let alive = self.backend.lock().await.is_server_alive().await;
        tracing::info!("[assistant] retrieve: backend lock released, alive={}", alive);
        if !alive {
            tracing::info!("[assistant] retrieve: server not alive, skipping RAG context");
            return Ok(vec![]);
        }
        tracing::info!("[assistant] retrieve: embedding query for top-{}", top_k);
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
