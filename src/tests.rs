#[cfg(test)]
mod unit_tests {
    use crate::{AssistantEngine, AssistantState, EngineConfig, RamTier};

    fn test_config(data_dir: &std::path::Path) -> EngineConfig {
        EngineConfig {
            data_dir: data_dir.to_path_buf(),
            model_id: "qwen3:test".into(),
            embedding_model: "nomic-embed-text".into(),
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

    // Regression test for the stale-model bug (D4): set_model_id/set_embedding_model
    // used to write straight into backend.config via try_lock and silently skip when
    // the backend was busy, so a model switch made mid-operation never took effect.
    // Every operation (provision::run, index_project, chat) now re-syncs
    // backend.config from inner.config under an AWAITED lock at call time — this
    // test exercises exactly that pattern (hold the backend lock to simulate "busy",
    // switch models, release, then sync-under-lock the way those call sites do) and
    // asserts the new model wins, not the one that was active when the lock was busy.
    #[tokio::test]
    async fn model_switch_while_backend_busy_is_picked_up_on_next_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        let engine = AssistantEngine::new(cfg);

        // Simulate the backend being busy (e.g. an in-flight chat/index) by holding
        // its lock on a spawned task while the model is switched.
        let backend = engine.backend.clone();
        let hold = backend.lock().await;
        engine.set_model_id("qwen3:new-model".into());
        engine.set_embedding_model("new-embed-model".into());
        drop(hold);

        // This mirrors the sync line provision::run()/index_project() perform right
        // after acquiring the backend lock.
        let config = engine.config();
        let mut b = backend.lock().await;
        b.config.model_id = config.model_id.clone();
        b.config.embedding_model = config.embedding_model.clone();

        assert_eq!(b.config.model_id, "qwen3:new-model");
        assert_eq!(b.config.embedding_model, "new-embed-model");
    }
}
