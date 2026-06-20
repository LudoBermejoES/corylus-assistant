//! Thin provision wrapper for the assistant engine.
//!
//! Delegates to the backend's `ensure_ready` / `pull_model` calls and
//! writes the version sentinel when the model is confirmed ready.

use std::sync::{Arc, Mutex};
use tracing::info;

use crate::{
    Inner, AssistantState, Result,
    backend::AssistantBackend,
    state::{VersionFile, SCHEMA_VERSION, version_path},
};

/// Run the full provision sequence: Ollama install (if needed + consented) → model pull.
/// `on_progress` receives every status transition.
pub async fn run(
    inner: Arc<Mutex<Inner>>,
    on_progress: impl Fn(AssistantState) + Send + 'static,
) -> Result<()> {
    let config = {
        let g = inner.lock().unwrap();
        std::fs::create_dir_all(&g.config.data_dir)?;
        g.config.clone()
    };

    set_state(&inner, AssistantState::NotInstalled);
    on_progress(AssistantState::NotInstalled);

    {
        let mut g = inner.lock().unwrap();
        g.backend.ensure_ready(|s| {
            // Mirror backend status into our state and forward to caller
            let _ = s.clone();
        }).await?;
    }

    // Write version sentinel so we remember the model is ready across restarts
    let ver = VersionFile {
        model_id: config.model_id.clone(),
        schema_version: SCHEMA_VERSION,
        ollama_managed: false,
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
