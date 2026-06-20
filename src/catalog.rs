/// A single entry in the curated Ollama model catalog.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    /// Ollama model tag (e.g. "qwen3:8b").
    pub id: &'static str,
    /// Human-readable name shown in the UI.
    pub display_name: &'static str,
    /// Role: "chat" or "embedding".
    pub role: &'static str,
    /// Approximate download size in gigabytes.
    pub size_gb: f32,
    /// Minimum recommended RAM in gigabytes for comfortable use.
    pub min_ram_gb: u32,
    /// One-sentence writer-friendly description.
    pub description: &'static str,
}

/// Curated catalog of supported Ollama models.
///
/// Chat models use the Qwen3 family (Apache 2.0); embedding model is nomic-embed-text (Apache 2.0).
/// Sizes are approximate GGUF Q4 download sizes. All models are offline after first pull.
pub const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        id: "qwen3:1.7b",
        display_name: "Qwen3 1.7B (Small)",
        role: "chat",
        size_gb: 1.1,
        min_ram_gb: 4,
        description: "Fastest model; ideal for 4–8 GB RAM machines. Good for simple questions.",
    },
    ModelEntry {
        id: "qwen3:8b",
        display_name: "Qwen3 8B (Balanced)",
        role: "chat",
        size_gb: 5.2,
        min_ram_gb: 10,
        description: "Best balance of quality and speed for 16 GB RAM machines.",
    },
    ModelEntry {
        id: "qwen3:14b",
        display_name: "Qwen3 14B (Large)",
        role: "chat",
        size_gb: 9.0,
        min_ram_gb: 20,
        description: "Highest quality responses; requires 20+ GB RAM.",
    },
    ModelEntry {
        id: "nomic-embed-text",
        display_name: "Nomic Embed Text",
        role: "embedding",
        size_gb: 0.27,
        min_ram_gb: 2,
        description: "Dedicated embedding model for project indexing. Small and fast.",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_chat_and_embedding_entries() {
        let chats: Vec<_> = CATALOG.iter().filter(|e| e.role == "chat").collect();
        let embeds: Vec<_> = CATALOG.iter().filter(|e| e.role == "embedding").collect();
        assert!(!chats.is_empty(), "catalog must have at least one chat model");
        assert!(!embeds.is_empty(), "catalog must have at least one embedding model");
    }

    #[test]
    fn catalog_entries_have_non_empty_fields() {
        for entry in CATALOG {
            assert!(!entry.id.is_empty(), "id must not be empty");
            assert!(!entry.display_name.is_empty(), "display_name must not be empty");
            assert!(!entry.description.is_empty(), "description must not be empty");
            assert!(entry.size_gb > 0.0, "size_gb must be positive");
        }
    }

    #[test]
    fn default_chat_model_in_catalog() {
        assert!(CATALOG.iter().any(|e| e.id == "qwen3:1.7b" && e.role == "chat"));
    }

    #[test]
    fn default_embedding_model_in_catalog() {
        assert!(CATALOG.iter().any(|e| e.id == "nomic-embed-text" && e.role == "embedding"));
    }
}
