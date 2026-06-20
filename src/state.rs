use std::path::PathBuf;
use crate::EngineConfig;

/// State machine for the assistant engine.
///
/// Transitions:
///   not_installed → needs_install (Ollama binary missing)
///   not_installed → server_down  (binary present, server not responding)
///   not_installed/server_down → downloading (model pull in progress)
///   downloading → indexing (embedding index build)
///   downloading/indexing → ready
///   any → error{message}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssistantState {
    NotInstalled,
    NeedsInstall,
    ServerDown,
    Downloading { downloaded: u64, total: Option<u64> },
    Indexing,
    Ready,
    Error { message: String },
}

pub fn assistant_dir(config: &EngineConfig) -> PathBuf {
    config.data_dir.clone()
}

pub fn version_path(config: &EngineConfig) -> PathBuf {
    config.data_dir.join("assistant.version.json")
}

pub fn part_path(config: &EngineConfig) -> PathBuf {
    config.data_dir.join("model.part")
}

pub fn index_dir(config: &EngineConfig) -> PathBuf {
    config.data_dir.join("index")
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct VersionFile {
    pub model_id: String,
    pub schema_version: u32,
    pub ollama_managed: bool,
}

pub const SCHEMA_VERSION: u32 = 1;

/// Returns true when Ollama has already been installed/managed by this engine
/// (a version file exists recording ollama_managed=true) OR the user has their
/// own Ollama and the configured model is present.
pub fn is_ready(config: &EngineConfig) -> bool {
    let ver = version_path(config);
    if !ver.exists() {
        return false;
    }
    let Ok(data) = std::fs::read_to_string(&ver) else { return false; };
    let Ok(v) = serde_json::from_str::<VersionFile>(&data) else { return false; };
    v.model_id == config.model_id && v.schema_version == SCHEMA_VERSION
}
