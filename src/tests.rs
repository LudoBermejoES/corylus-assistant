#[cfg(test)]
mod tests {
    use crate::{AssistantEngine, AssistantState, EngineConfig, RamTier};

    fn test_config(data_dir: &std::path::Path) -> EngineConfig {
        EngineConfig {
            data_dir: data_dir.to_path_buf(),
            model_id: "qwen3:test".into(),
            embedding_model: "qwen3:test".into(),
            ollama_endpoint: "http://127.0.0.1:11434".into(),
            context_window: 4096,
        }
    }

    #[test]
    fn initial_state_is_not_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        let engine = AssistantEngine::new(cfg);
        assert_eq!(engine.state(), AssistantState::NotInstalled);
    }

    #[test]
    fn set_data_dir_updates_path() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = EngineConfig::default_for_ram(
            std::path::PathBuf::from("__unconfigured__"),
            None,
        );
        let engine = AssistantEngine::new(cfg);
        let new_dir = tmp.path().join("assistant");
        engine.set_data_dir(new_dir.clone());
        assert_eq!(engine.data_dir(), new_dir);
    }

    #[test]
    fn ram_tier_low_picks_small_model() {
        let tier = RamTier::from_bytes(Some(4 * 1024 * 1024 * 1024));
        assert_eq!(tier, RamTier::Low);
        assert_eq!(tier.default_model(), "qwen3:1.7b");
    }

    #[test]
    fn ram_tier_medium() {
        let tier = RamTier::from_bytes(Some(16 * 1024 * 1024 * 1024));
        assert_eq!(tier, RamTier::Medium);
        assert_eq!(tier.default_model(), "qwen3:8b");
    }

    #[test]
    fn ram_tier_high() {
        let tier = RamTier::from_bytes(Some(32 * 1024 * 1024 * 1024));
        assert_eq!(tier, RamTier::High);
        assert_eq!(tier.default_model(), "qwen3:14b");
    }

    #[test]
    fn uninstall_removes_version_sentinel() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        // Write a fake version sentinel
        let ver_path = tmp.path().join("assistant.version.json");
        std::fs::write(&ver_path, r#"{"model_id":"qwen3:test","schema_version":1,"ollama_managed":false}"#).unwrap();
        let engine = AssistantEngine::new(cfg);
        engine.uninstall().unwrap();
        assert!(!ver_path.exists());
        assert_eq!(engine.state(), AssistantState::NotInstalled);
    }
}
