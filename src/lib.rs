//! Local LLM assistant engine for Corylus.
//!
//! Architecture mirrors the other Corylus vendor crates (rust-pos, rust-lemma, etc.):
//!   - All assistant logic lives here; the app contains only thin Tauri command glue.
//!   - Download-on-demand: Ollama + a Qwen3 model are fetched at runtime, never bundled.
//!   - Backend abstraction: `AssistantBackend` trait; Phase 1 = `OllamaBackend`.
//!   - State machine: not_installed → needs_install/server_down → downloading → ready | error.
//!   - RAG grounding: sqlite-vec index over manuscript + story-bible (stubs for Phase 2).

mod error;
mod state;
mod backend;
mod ollama;
mod provision;

#[cfg(test)]
mod tests;

pub use error::AssistantError;
pub use state::AssistantState;
pub use backend::{AssistantBackend, ChatMessage, ChatToken, RamTier, RetrievedChunk, build_system_prompt};
pub use ollama::OllamaBackend;

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
            b.config.data_dir = data_dir;
            let s = b.probe();
            g.state = s;
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
