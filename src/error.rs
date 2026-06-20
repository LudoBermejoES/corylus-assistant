use thiserror::Error;

#[derive(Debug, Error)]
pub enum AssistantError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Ollama is not installed")]
    OllamaNotInstalled,

    #[error("Ollama server is not running")]
    OllamaServerDown,

    #[error("Ollama install failed: {0}")]
    OllamaInstallFailed(String),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),

    #[error("Request cancelled")]
    Cancelled,

    #[error("Context too large")]
    ContextTooLarge,

    #[error("Embedding model mismatch: index was built with model '{index_model}' ({index_dim} dims) but configured model is '{config_model}'")]
    EmbeddingModelMismatch {
        index_model: String,
        index_dim: u32,
        config_model: String,
    },

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<&str> for AssistantError {
    fn from(s: &str) -> Self {
        AssistantError::Internal(s.to_string())
    }
}
