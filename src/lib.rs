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
pub mod gpu;
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
/// Turns of history a chat request carries, and the most characters of each.
pub const HISTORY_TURNS: usize = 4;
pub const HISTORY_TURN_CHARS: usize = 1000;

pub use backend::{AssistantBackend, ChatMessage, ChatToken, RamTier, RetrievedChunk, build_system_prompt};
pub use ollama::{OllamaBackend, detect_system_ram};
pub use index::{VectorIndex, RetrievedRow};
pub use catalog::{ModelEntry, CATALOG};
pub use fallback::{FallbackEngine, FsmState, FsmFlow};
pub use router::{Router, RouterResponse, is_llm_ready};
pub use gpu::{Backend as GpuBackend, GpuStatus};
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
    /// True while a GPU add-on install or backend switch runs.
    gpu_busy: Arc<AtomicBool>,
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
            gpu_busy: Arc::new(AtomicBool::new(false)),
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
        // An installed assistant is only "ready" once its server answers:
        // `ensure_server`/`refresh_status` promote it to Ready after verifying.
        if state::is_ready(&g.config) {
            g.state = AssistantState::ServerDown;
        }
    }

    /// True when the assistant was installed (version sentinel matches the model).
    pub fn is_installed(&self) -> bool {
        state::is_ready(&self.config())
    }

    /// Make sure the Ollama server answers, starting it (with the selected GPU
    /// backend) when it is down, and keep the state honest: Ready on success,
    /// Error with the reason on failure. Only touches Ready/ServerDown/Error.
    pub async fn ensure_server(&self) -> Result<()> {
        let b = self.backend.lock().await;
        self.start_locked(b).await
    }

    async fn start_locked(&self, mut b: tokio::sync::MutexGuard<'_, OllamaBackend>) -> Result<()> {
        let res = b.ensure_server_running().await;
        drop(b);
        let installed = self.is_installed();
        let mut g = self.inner.lock().unwrap();
        if matches!(g.state, AssistantState::Ready | AssistantState::ServerDown | AssistantState::Error { .. }) {
            match &res {
                Ok(()) if installed => g.state = AssistantState::Ready,
                Err(e) if installed => g.state = AssistantState::Error { message: e.to_string() },
                _ => {}
            }
        }
        res
    }

    /// Current state, verified: an installed assistant whose server is down is
    /// started first. Skipped (current state returned) while another operation
    /// such as a model pull holds the backend.
    pub async fn refresh_status(&self) -> AssistantState {
        if self.is_installed()
            && matches!(self.state(), AssistantState::Ready | AssistantState::ServerDown | AssistantState::Error { .. })
        {
            if let Ok(b) = self.backend.try_lock() {
                let _ = self.start_locked(b).await;
            }
        }
        self.state()
    }

    /// Where `ollama-server.log` goes (the app log dir). Called from app setup.
    pub fn set_log_dir(&self, log_dir: PathBuf) {
        if let Ok(mut b) = self.backend.try_lock() {
            b.log_dir = Some(log_dir);
        }
    }

    // ── GPU acceleration (see gpu.rs) ──────────────────────────────────────────

    /// Approximate size of the configured chat model, to judge whether a card fits it.
    fn chat_model_bytes(&self) -> u64 {
        let id = self.config().model_id;
        CATALOG
            .iter()
            .find(|e| e.id == id)
            .map(|e| (e.size_gb as f64 * 1e9) as u64)
            .unwrap_or_else(|| RamTier::from_bytes(detect_system_ram()).default_model_size_bytes())
    }

    /// True when Corylus installed Ollama itself (only then may it add GPU support).
    fn managed_install_present(&self) -> bool {
        let bin = if cfg!(target_os = "windows") { "ollama.exe" } else { "ollama" };
        self.data_dir().join("bin").join(bin).exists()
    }

    /// The graphics card, the backend in use and the add-on that would help.
    pub fn gpu_status(&self) -> GpuStatus {
        let data_dir = self.data_dir();
        gpu::status(
            self.managed_install_present(),
            &gpu::detect_adapters(),
            self.chat_model_bytes(),
            gpu::load_state(&data_dir).as_ref(),
        )
    }

    /// Download (with consent, given in the UI) and install the recommended GPU
    /// add-on, then restart the server on it and verify. `on_progress` gets
    /// (downloaded, total) bytes. Cancel with `cancel_pull()`.
    pub async fn gpu_install(&self, on_progress: impl Fn(u64, u64) + Send + Sync) -> Result<GpuStatus> {
        if self.gpu_busy.swap(true, Ordering::SeqCst) {
            return Err(AssistantError::Internal("graphics support is already being set up".into()));
        }
        let result = self.gpu_install_inner(on_progress).await;
        self.gpu_busy.store(false, Ordering::SeqCst);
        result
    }

    async fn gpu_install_inner(&self, on_progress: impl Fn(u64, u64) + Send + Sync) -> Result<GpuStatus> {
        let status = self.gpu_status();
        if !status.supported {
            return Err(AssistantError::OllamaInstallFailed(
                "graphics support needs the assistant installed by Corylus on Windows".into(),
            ));
        }
        let backend = status.recommended;
        let addon = gpu::addon(backend).ok_or_else(|| {
            AssistantError::OllamaInstallFailed("no usable graphics card for the assistant".into())
        })?;
        let data_dir = self.data_dir();
        let installed = gpu::load_state(&data_dir).is_some_and(|s| s.installed.contains(&backend));

        // Download without the backend lock, so chat keeps working meanwhile.
        self.cancel_pull.store(false, Ordering::Relaxed);
        let part = if installed {
            None
        } else {
            Some(gpu::download_addon(addon, &data_dir, &self.cancel_pull, on_progress).await?)
        };

        let mut b = self.backend.lock().await;
        b.engine_managed_install = true;
        if let Some(part) = part {
            b.stop_managed_server().await;
            let bin_dir = data_dir.join("bin");
            let zip = part.clone();
            let res = tokio::task::spawn_blocking(move || gpu::install_addon_zip(&zip, &bin_dir, backend))
                .await
                .map_err(|e| AssistantError::Internal(e.to_string()));
            let _ = std::fs::remove_file(&part);
            res??;
        }
        let mut state = gpu::load_state(&data_dir).unwrap_or_default();
        if !state.installed.contains(&backend) {
            state.installed.push(backend);
        }
        state.active = Some(backend);
        state.cpu_only = false;
        gpu::save_state(&data_dir, &state)?;
        b.restart_and_verify().await?;
        drop(b);
        Ok(self.gpu_status())
    }

    /// Run on the processor (`true`) or go back to the installed graphics
    /// support (`false`, which also forgets earlier failures and retries).
    pub async fn gpu_set_cpu_only(&self, cpu_only: bool) -> Result<GpuStatus> {
        if self.gpu_busy.swap(true, Ordering::SeqCst) {
            return Err(AssistantError::Internal("graphics support is already being set up".into()));
        }
        let result = self.gpu_set_cpu_only_inner(cpu_only).await;
        self.gpu_busy.store(false, Ordering::SeqCst);
        result
    }

    async fn gpu_set_cpu_only_inner(&self, cpu_only: bool) -> Result<GpuStatus> {
        let data_dir = self.data_dir();
        let mut state = gpu::load_state(&data_dir)
            .filter(|s| !s.installed.is_empty())
            .ok_or_else(|| AssistantError::Internal("no graphics support installed".into()))?;
        state.cpu_only = cpu_only;
        if !cpu_only {
            state.failed.clear();
            let adapters = gpu::detect_adapters();
            let best = gpu::best_adapter(&adapters, self.chat_model_bytes());
            let wanted = gpu::choose(best, &[]);
            state.active = Some(if state.installed.contains(&wanted) {
                wanted
            } else {
                [gpu::Backend::Cuda, gpu::Backend::Vulkan]
                    .into_iter()
                    .find(|b| state.installed.contains(b))
                    .unwrap_or(gpu::Backend::Cpu)
            });
        }
        gpu::save_state(&data_dir, &state)?;
        let mut b = self.backend.lock().await;
        b.engine_managed_install = true;
        b.restart_and_verify().await?;
        drop(b);
        Ok(self.gpu_status())
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

    /// Override the chat model ID. `inner.config` is the single source of truth;
    /// every operation (chat/provision/index_project) syncs the backend's config
    /// from it under the backend lock at call time, so there is no race with the
    /// backend being busy at selection time.
    pub fn set_model_id(&self, model_id: String) {
        self.inner.lock().unwrap().config.model_id = model_id;
    }

    /// Override the embedding model ID. See `set_model_id` — same single-source-of-truth pattern.
    pub fn set_embedding_model(&self, embedding_model: String) {
        self.inner.lock().unwrap().config.embedding_model = embedding_model;
    }

    /// Begin provisioning (install Ollama + pull model). Consent must be given before calling.
    pub async fn provision(
        &self,
        on_progress: impl Fn(AssistantState) + Send + Sync + 'static,
    ) -> Result<()> {
        self.cancel_pull.store(false, Ordering::Relaxed);
        provision::run(self.inner.clone(), self.backend.clone(), self.cancel_pull.clone(), on_progress).await
    }

    /// Detect available system RAM (best-effort). Awaits the backend lock rather
    /// than `try_lock`-and-degrade, so a concurrent operation holding the lock
    /// delays this call instead of causing it to silently report no RAM detected.
    pub async fn detect_ram_bytes(&self) -> Option<u64> {
        self.backend.lock().await.detect_ram_bytes()
    }

    /// Return the RAM-aware default model tag and its approximate size/RAM.
    pub async fn model_info(&self) -> ModelInfo {
        let ram = self.detect_ram_bytes().await;
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
            // Sync embedding_model from inner.config into the backend at call time —
            // the same pattern chat() uses for model_id — so a model switch made via
            // set_embedding_model() while the backend was busy (try_lock failed) still
            // takes effect here rather than silently indexing with the previous model.
            b.config.embedding_model = config.embedding_model.clone();
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
        idx.chunk_count()
    }

    /// Send a chat message grounded in the RAG index and return the full response text.
    /// Retrieves the top-5 relevant chunks, builds a system prompt, and calls the backend.
    pub async fn chat(&self, user_message: &str) -> Result<String> {
        self.chat_with(user_message, None, &[]).await
    }

    /// `chat` with a companion `persona` (voice only, inside the fixed rules) and the last turns
    /// of the conversation. Only `user` and `assistant` turns are kept, the last
    /// [`HISTORY_TURNS`] of them: the caller cannot smuggle in a `system` message.
    pub async fn chat_with(&self, user_message: &str, persona: Option<&str>, history: &[ChatMessage]) -> Result<String> {
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
        let system_prompt = backend::build_system_prompt(&chunks, persona, fallback::detect_lang(user_message));

        let turns: Vec<&ChatMessage> = history.iter().filter(|m| m.role == "user" || m.role == "assistant").collect();
        let mut messages = vec![ChatMessage { role: "system".into(), content: system_prompt }];
        messages.extend(turns[turns.len().saturating_sub(HISTORY_TURNS)..].iter().map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.chars().take(HISTORY_TURN_CHARS).collect(),
        }));
        messages.push(ChatMessage { role: "user".into(), content: user_message.into() });

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
