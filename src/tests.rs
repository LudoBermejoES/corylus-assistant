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
    fn ram_tier_boundaries_and_unknown() {
        const GB: u64 = 1024 * 1024 * 1024;
        assert_eq!(RamTier::from_bytes(None), RamTier::Low);
        assert_eq!(RamTier::from_bytes(Some(8 * GB)), RamTier::Low);
        assert_eq!(RamTier::from_bytes(Some(8 * GB + 1)), RamTier::Medium);
        assert_eq!(RamTier::from_bytes(Some(20 * GB)), RamTier::Medium);
        assert_eq!(RamTier::from_bytes(Some(56 * GB)), RamTier::High);
    }

    /// Detection must work on every desktop OS: a 0/None here made a 56 GB
    /// Windows machine report "RAM tier: low".
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    #[test]
    fn system_ram_is_detected_on_desktop() {
        let ram = crate::detect_system_ram().expect("RAM not detected");
        assert!(ram > 512 * 1024 * 1024, "implausible RAM: {ram}");
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

#[cfg(test)]
mod server_tests {
    use crate::{AssistantEngine, AssistantState, EngineConfig, Router};
    use std::time::Duration;

    /// Minimal HTTP server answering 200 `{"models":[]}` to every request.
    async fn fake_server() -> (String, tokio::task::JoinHandle<()>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let h = tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = l.accept().await else { return };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 2048];
                    let _ = s.read(&mut buf).await;
                    let body = r#"{"models":[]}"#;
                    let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
                });
            }
        });
        (url, h)
    }

    /// Installed assistant (version sentinel written) pointing at `endpoint`.
    fn installed_engine(dir: &std::path::Path, endpoint: &str) -> AssistantEngine {
        let cfg = EngineConfig {
            data_dir: dir.to_path_buf(),
            model_id: "qwen3:test".into(),
            embedding_model: "nomic-embed-text".into(),
            ollama_endpoint: endpoint.into(),
            context_window: 4096,
        };
        std::fs::write(
            dir.join("assistant.version.json"),
            r#"{"model_id":"qwen3:test","schema_version":1,"ollama_managed":true}"#,
        )
        .unwrap();
        let e = AssistantEngine::new(cfg);
        e.set_data_dir(dir.to_path_buf());
        e.backend.try_lock().unwrap().server_start_timeout = Duration::from_secs(2);
        e
    }

    /// A port nothing listens on.
    async fn dead_endpoint() -> String {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        format!("http://{}", l.local_addr().unwrap())
    }

    #[tokio::test]
    async fn restart_is_not_ready_until_server_verified() {
        let tmp = tempfile::tempdir().unwrap();
        let e = installed_engine(tmp.path(), &dead_endpoint().await);
        assert_eq!(e.state(), AssistantState::ServerDown);
    }

    #[tokio::test]
    async fn refresh_promotes_to_ready_when_server_answers() {
        let tmp = tempfile::tempdir().unwrap();
        let (url, h) = fake_server().await;
        let e = installed_engine(tmp.path(), &url);
        assert_eq!(e.refresh_status().await, AssistantState::Ready);
        h.abort();
    }

    #[tokio::test]
    async fn failed_start_reports_error_not_ready() {
        let tmp = tempfile::tempdir().unwrap();
        // No ollama binary in <data_dir>/bin (and none on PATH in CI): start must fail.
        let e = installed_engine(tmp.path(), &dead_endpoint().await);
        assert!(e.ensure_server().await.is_err());
        assert!(matches!(e.state(), AssistantState::Error { .. }));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn server_that_exits_at_once_fails_fast_with_reason() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let exe = bin.join("ollama");
        std::fs::write(&exe, "#!/bin/sh\nexit 3\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        let e = installed_engine(tmp.path(), &dead_endpoint().await);
        e.backend.try_lock().unwrap().server_start_timeout = Duration::from_secs(20);
        let t = std::time::Instant::now();
        let err = e.ensure_server().await.unwrap_err().to_string();
        assert!(err.contains("exited"), "{err}");
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn chat_tags_basic_mode_when_server_cannot_start() {
        let tmp = tempfile::tempdir().unwrap();
        let e = installed_engine(tmp.path(), &dead_endpoint().await);
        let r = Router::new().chat("hello", &mut None, &e, &[]).await;
        assert_eq!(r.source, "llm_unavailable");
    }

    #[tokio::test]
    async fn chat_without_install_stays_plain_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = EngineConfig::default_for_ram(tmp.path().to_path_buf(), None);
        let e = AssistantEngine::new(cfg);
        let r = Router::new().chat("hello", &mut None, &e, &[]).await;
        assert_eq!(r.source, "fallback");
    }
}
