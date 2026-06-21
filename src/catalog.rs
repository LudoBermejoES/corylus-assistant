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
/// Chat models: Qwen3 (Apache 2.0), Gemma 3 (Gemma ToU), Llama 3.2 (Meta Community License),
/// Mistral Small (Apache 2.0), DeepSeek-R1 distills (MIT/Apache 2.0).
/// Embedding: nomic-embed-text (Apache 2.0), mxbai-embed-large (Apache 2.0).
/// Sizes are approximate GGUF Q4_K_M download sizes. All models run offline after first pull.
pub const CATALOG: &[ModelEntry] = &[
    // ── Qwen3 family (Alibaba, released April 2025) ──────────────────────────
    ModelEntry {
        id: "qwen3:1.7b",
        display_name: "Qwen3 1.7B",
        role: "chat",
        size_gb: 1.1,
        min_ram_gb: 4,
        description: "Fastest response; great for quick questions on 4–8 GB RAM machines.",
    },
    ModelEntry {
        id: "qwen3:4b",
        display_name: "Qwen3 4B",
        role: "chat",
        size_gb: 2.6,
        min_ram_gb: 6,
        description: "Good quality on modest hardware; solid choice for 8 GB RAM machines.",
    },
    ModelEntry {
        id: "qwen3:8b",
        display_name: "Qwen3 8B",
        role: "chat",
        size_gb: 5.2,
        min_ram_gb: 10,
        description: "Best balance of quality and speed; recommended for 16 GB RAM.",
    },
    ModelEntry {
        id: "qwen3:14b",
        display_name: "Qwen3 14B",
        role: "chat",
        size_gb: 9.0,
        min_ram_gb: 20,
        description: "High quality writing assistance; needs 20+ GB RAM.",
    },
    // ── Gemma 3 family (Google, released March 2025) ─────────────────────────
    ModelEntry {
        id: "gemma3:4b",
        display_name: "Gemma 3 4B",
        role: "chat",
        size_gb: 3.3,
        min_ram_gb: 6,
        description: "Google's compact model; strong instruction following on 8 GB RAM.",
    },
    ModelEntry {
        id: "gemma3:12b",
        display_name: "Gemma 3 12B",
        role: "chat",
        size_gb: 8.1,
        min_ram_gb: 14,
        description: "Google's mid-size model; excellent writing quality for 16 GB RAM.",
    },
    // ── Llama 3.2 family (Meta, released September 2024) ────────────────────
    ModelEntry {
        id: "llama3.2:3b",
        display_name: "Llama 3.2 3B",
        role: "chat",
        size_gb: 2.0,
        min_ram_gb: 6,
        description: "Meta's lightweight model; fast and capable on any modern laptop.",
    },
    // ── Mistral Small (Mistral AI, released January 2025) ────────────────────
    ModelEntry {
        id: "mistral-small3.1:24b",
        display_name: "Mistral Small 3.1 24B",
        role: "chat",
        size_gb: 14.0,
        min_ram_gb: 24,
        description: "Mistral's latest compact model; near-frontier quality for 32 GB RAM machines.",
    },
    // ── DeepSeek-R1 distills (DeepSeek, released January 2025) ──────────────
    ModelEntry {
        id: "deepseek-r1:7b",
        display_name: "DeepSeek R1 7B",
        role: "chat",
        size_gb: 4.7,
        min_ram_gb: 8,
        description: "Reasoning-focused model; thinks step by step before answering. Good for 16 GB RAM.",
    },
    ModelEntry {
        id: "deepseek-r1:14b",
        display_name: "DeepSeek R1 14B",
        role: "chat",
        size_gb: 9.0,
        min_ram_gb: 16,
        description: "Stronger reasoning distill; excellent for complex creative problems on 24 GB RAM.",
    },
    // ── Embedding models ─────────────────────────────────────────────────────
    ModelEntry {
        id: "nomic-embed-text",
        display_name: "Nomic Embed Text",
        role: "embedding",
        size_gb: 0.27,
        min_ram_gb: 2,
        description: "Fast, lightweight embedding model for project indexing. Recommended default.",
    },
    ModelEntry {
        id: "mxbai-embed-large",
        display_name: "MixedBread Embed Large",
        role: "embedding",
        size_gb: 0.67,
        min_ram_gb: 2,
        description: "Higher accuracy embeddings; better search results at a small size cost.",
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

    #[test]
    fn all_model_roles_are_chat_or_embedding() {
        for entry in CATALOG {
            assert!(
                entry.role == "chat" || entry.role == "embedding",
                "unexpected role '{}' for model '{}'",
                entry.role,
                entry.id,
            );
        }
    }

    #[test]
    fn catalog_contains_recent_models() {
        let ids: Vec<&str> = CATALOG.iter().map(|e| e.id).collect();
        // Models added in the 2025 catalog expansion
        assert!(ids.contains(&"gemma3:4b"), "gemma3:4b must be in catalog");
        assert!(ids.contains(&"gemma3:12b"), "gemma3:12b must be in catalog");
        assert!(ids.contains(&"llama3.2:3b"), "llama3.2:3b must be in catalog");
        assert!(ids.contains(&"deepseek-r1:7b"), "deepseek-r1:7b must be in catalog");
        assert!(ids.contains(&"deepseek-r1:14b"), "deepseek-r1:14b must be in catalog");
        assert!(ids.contains(&"mxbai-embed-large"), "mxbai-embed-large must be in catalog");
    }

    #[test]
    fn catalog_ram_requirements_are_positive() {
        for entry in CATALOG {
            assert!(
                entry.min_ram_gb >= 2,
                "min_ram_gb for '{}' should be at least 2 GB, got {}",
                entry.id,
                entry.min_ram_gb,
            );
        }
    }

    #[test]
    fn chat_models_are_larger_than_embedding_models() {
        let max_embed = CATALOG
            .iter()
            .filter(|e| e.role == "embedding")
            .map(|e| e.size_gb)
            .fold(0.0_f32, f32::max);
        let min_chat = CATALOG
            .iter()
            .filter(|e| e.role == "chat")
            .map(|e| e.size_gb)
            .fold(f32::MAX, f32::min);
        assert!(
            min_chat > max_embed,
            "smallest chat model ({} GB) should be larger than largest embedding model ({} GB)",
            min_chat,
            max_embed,
        );
    }
}
