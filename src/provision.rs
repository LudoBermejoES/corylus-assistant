//! Thin provision wrapper for the assistant engine.
//!
//! Delegates to the backend's `ensure_ready` / `pull_model` calls and
//! writes the version sentinel when the model is confirmed ready.

use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicBool;
use tracing::info;

use crate::{
    Inner, AssistantState, Result,
    backend::AssistantBackend,
    ollama::OllamaBackend,
    state::{VersionFile, SCHEMA_VERSION, version_path},
};

/// Run the full provision sequence: Ollama install (if needed + consented) → model pull.
/// `on_progress` receives every status transition.
///
/// `backend` is a separate `tokio::sync::Mutex` so we can `.await` inside it
/// without violating the `Send` bound — `std::sync::MutexGuard` is not `Send`.
pub async fn run(
    inner: Arc<Mutex<Inner>>,
    backend: Arc<tokio::sync::Mutex<OllamaBackend>>,
    cancel: Arc<AtomicBool>,
    on_progress: impl Fn(AssistantState) + Send + Sync + 'static,
) -> Result<()> {
    let config = {
        let g = inner.lock().unwrap();
        std::fs::create_dir_all(&g.config.data_dir)?;
        g.config.clone()
    };

    set_state(&inner, AssistantState::NotInstalled);
    on_progress(AssistantState::NotInstalled);

    // Wrap on_progress in Arc so both the ensure_ready closure and the post-await
    // call share the same callback without moving it.
    let on_progress = std::sync::Arc::new(on_progress);
    let on_progress2 = on_progress.clone();
    let inner2 = inner.clone();
    {
        let mut b = backend.lock().await;
        b.set_cancel(cancel);
        b.ensure_ready(move |s| {
            inner2.lock().unwrap().state = s.clone();
            on_progress2(s);
        }).await?;
    }

    // Write version sentinel so we remember the model is ready across restarts.
    // Capture whether we managed the Ollama install so we know to set OLLAMA_MODELS on next start.
    let ollama_managed = backend.lock().await.engine_managed_install;
    let ver = VersionFile {
        model_id: config.model_id.clone(),
        schema_version: SCHEMA_VERSION,
        ollama_managed,
    };
    let ver_path = version_path(&config);
    std::fs::write(&ver_path, serde_json::to_string_pretty(&ver)?)?;

    set_state(&inner, AssistantState::Ready);
    on_progress(AssistantState::Ready);
    info!("[assistant] provision complete, model {}", config.model_id);
    Ok(())
}

pub fn set_state(inner: &Arc<Mutex<Inner>>, state: AssistantState) {
    inner.lock().unwrap().state = state;
}
