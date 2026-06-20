//! Ollama HTTP backend implementation.
//!
//! Probes for an existing Ollama server, starts one if the binary is present,
//! or provisions a private copy into `<data_dir>/bin/` from a vendored zip.
//! All inference traffic stays on 127.0.0.1:11434.
//!
//! # Install strategy
//!
//! We never run `install.sh`, never call `sudo`, and never touch system paths.
//! Instead, `provision_ollama()` downloads a pre-built platform zip (produced by
//! `scripts/prepare_ollama.py` and uploaded to a GitHub Release), verifies its
//! SHA-256, and extracts the files into `<data_dir>/bin/`.  The Ollama server is
//! then started with `OLLAMA_MODELS=<data_dir>/models` so all model weights also
//! live under the app's own data directory and uninstall is a simple `rm -rf`.
//!
//! If the user already has a system Ollama on PATH we reuse it as-is (no model
//! relocation — they own that install).

use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender as Sender;
use tracing::{info, warn};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};

use crate::{
    AssistantState, EngineConfig, Result,
    backend::{AssistantBackend, ChatMessage, ChatToken},
    error::AssistantError,
};

// ── Asset constants ────────────────────────────────────────────────────────────

const OLLAMA_VERSION: &str = "v0.30.10-1";

// SHA-256 of the repacked zips produced by scripts/prepare_ollama.py.
// Update these (and OLLAMA_VERSION above) whenever the script is re-run.
const OLLAMA_SHA256_MACOS_UNIVERSAL: &str =
    "11f889a1ab3c4b3f91561a3b54b2438a30029c271aa07da22e6f38e511fe4781";
const OLLAMA_SHA256_LINUX_X86_64: &str =
    "d6109b0be28b4b285ca49b435c82941f6d04cdfaadc08fa512e40a03668bb10a";
const OLLAMA_SHA256_LINUX_AARCH64: &str =
    "8602ed350edf273d07e357aa60c328d5b6d4a79111bb1949987aa9f2bf738489";
const OLLAMA_SHA256_WINDOWS_X86_64: &str =
    "a037da7ff3f49f7dfc2431a42686b3027a75ecb3e6684276e6b9f15938fb37cd";

// Base URL for the vendored zips on the rust-assistant GitHub Release.
const ASSET_BASE_URL: &str =
    "https://github.com/LudoBermejoES/corylus-assistant/releases/download";
// GitHub Release tag for the asset zips (may differ from OLLAMA_VERSION when
// we repack with fixes without bumping the upstream Ollama version).
const ASSET_RELEASE_TAG: &str = "v0.30.10";

// ── Misc constants ─────────────────────────────────────────────────────────────

const OLLAMA_BASE: &str = "http://127.0.0.1:11434";
const HEALTH_TIMEOUT_SECS: u64 = 3;
/// Name of the version marker file written after a successful private install.
const VERSION_FILE: &str = "ollama-version.json";

// ── OllamaBackend ──────────────────────────────────────────────────────────────

pub struct OllamaBackend {
    pub config: EngineConfig,
    pub state: AssistantState,
    client: reqwest::Client,
    /// True when Corylus provisioned its own private Ollama binary.
    /// In that case we set OLLAMA_MODELS so weights stay under our data dir.
    pub(crate) engine_managed_install: bool,
}

impl OllamaBackend {
    pub fn new(config: EngineConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("HTTP client build");
        Self {
            state: AssistantState::NotInstalled,
            config,
            client,
            engine_managed_install: false,
        }
    }

    // ── Server health ──────────────────────────────────────────────────────────

    async fn server_alive(&self) -> bool {
        let alive = self.client
            .get(format!("{}/api/tags", OLLAMA_BASE))
            .timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        info!("[assistant] server_alive: {}", alive);
        alive
    }

    /// Pull the embedding model if it is not yet present. Safe to call on every index.
    pub(crate) async fn ensure_embedding_model(&self) -> Result<()> {
        let embed_id = self.config.embedding_model.clone();
        if self.model_present(&embed_id).await {
            return Ok(());
        }
        info!("[assistant] embedding model {} not present, pulling now", embed_id);
        let resp = self
            .client
            .post(format!("{}/api/pull", OLLAMA_BASE))
            .json(&serde_json::json!({ "name": embed_id, "stream": true }))
            .send()
            .await
            .map_err(AssistantError::Http)?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(AssistantError::Http)?;
            for line in chunk.split(|&b| b == b'\n') {
                if line.is_empty() { continue; }
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(line) {
                    if val.get("status").and_then(|v| v.as_str()) == Some("success") {
                        break;
                    }
                }
            }
        }
        info!("[assistant] embedding model {} ready", embed_id);
        Ok(())
    }

    pub(crate) async fn ensure_server_running(&mut self) -> Result<()> {
        info!("[assistant] ensure_server_running: checking if server is alive");
        if self.server_alive().await {
            info!("[assistant] ensure_server_running: server already alive");
            return Ok(());
        }
        info!("[assistant] ensure_server_running: server not alive, engine_managed_install={}", self.engine_managed_install);

        let bin = self.ollama_binary_path();
        let Some(bin_path) = bin else {
            warn!("[assistant] ensure_server_running: no ollama binary found");
            return Err(AssistantError::OllamaNotInstalled);
        };

        info!("[assistant] starting ollama serve from {}", bin_path.display());
        let mut cmd = tokio::process::Command::new(&bin_path);
        cmd.arg("serve")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        if self.engine_managed_install {
            let models_dir = self.config.data_dir.join("models");
            cmd.env("OLLAMA_MODELS", &models_dir);
            // Also put the private bin dir on PATH so dylib/so resolution works
            let bin_dir = self.config.data_dir.join("bin");
            prepend_path_env(&mut cmd, &bin_dir);
            info!("[assistant] OLLAMA_MODELS={}", models_dir.display());
        }

        match cmd.spawn() {
            Ok(_) => info!("[assistant] ollama serve spawned"),
            Err(e) => {
                warn!("[assistant] failed to spawn ollama serve: {}", e);
                return Err(AssistantError::OllamaServerDown);
            }
        }

        // Wait up to 8 s for the server to come up
        for i in 0..16 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if self.server_alive().await {
                info!("[assistant] ollama server started (attempt {})", i + 1);
                return Ok(());
            }
        }
        warn!("[assistant] ollama serve did not start in time after 8s");
        Err(AssistantError::OllamaServerDown)
    }

    // ── Binary detection ───────────────────────────────────────────────────────

    /// Returns the path to use for `ollama` invocations, or `None` if neither
    /// the private copy nor a system install is present.
    ///
    /// Priority:
    ///   1. Private copy in `<data_dir>/bin/ollama[.exe]`  (managed by Corylus)
    ///   2. System Ollama on PATH                           (user's own install)
    fn ollama_binary_path(&self) -> Option<PathBuf> {
        let private = self.private_bin_path();
        if private.exists() {
            return Some(private);
        }
        which::which("ollama").ok()
    }

    fn private_bin_path(&self) -> PathBuf {
        let bin_name = if cfg!(target_os = "windows") { "ollama.exe" } else { "ollama" };
        self.config.data_dir.join("bin").join(bin_name)
    }

    fn ollama_binary_present(&self) -> bool {
        self.ollama_binary_path().is_some()
    }

    fn private_install_present(&self) -> bool {
        // Check the version marker so we know the install is complete, not partial.
        let marker = self.config.data_dir.join(VERSION_FILE);
        if !marker.exists() {
            return false;
        }
        let Ok(json) = std::fs::read_to_string(&marker) else { return false; };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) else { return false; };
        v.get("version").and_then(|s| s.as_str()) == Some(OLLAMA_VERSION)
    }

    // ── Model presence ─────────────────────────────────────────────────────────

    async fn model_present(&self, model_id: &str) -> bool {
        let Ok(resp) = self
            .client
            .get(format!("{}/api/tags", OLLAMA_BASE))
            .send()
            .await
        else {
            return false;
        };
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return false;
        };
        body.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter().any(|m| {
                    m.get("name")
                        .and_then(|n| n.as_str())
                        .map(|n| {
                            n == model_id
                                || n.starts_with(&format!(
                                    "{}:",
                                    model_id.split(':').next().unwrap_or(model_id)
                                ))
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    // ── Provision ──────────────────────────────────────────────────────────────

    async fn provision_ollama(
        &mut self,
        on_status: &(dyn Fn(AssistantState) + Sync),
    ) -> Result<()> {
        if self.private_install_present() {
            info!("[assistant] private Ollama {} already provisioned", OLLAMA_VERSION);
            self.engine_managed_install = true;
            return Ok(());
        }

        let (asset_name, sha256_expected) = platform_asset()?;
        let url = format!("{}/{}/{}", ASSET_BASE_URL, ASSET_RELEASE_TAG, asset_name);
        let bin_dir = self.config.data_dir.join("bin");
        tokio::fs::create_dir_all(&bin_dir).await?;

        info!("[assistant] downloading Ollama from {}", url);

        // ── Download with progress ────────────────────────────────────────────
        let part_path = self.config.data_dir.join("ollama.part");
        {
            let resp = self
                .client
                .get(&url)
                .timeout(Duration::from_secs(600))
                .send()
                .await
                .map_err(AssistantError::Http)?
                .error_for_status()
                .map_err(AssistantError::Http)?;

            let total = resp.content_length();
            let mut downloaded: u64 = 0;
            let mut stream = resp.bytes_stream();
            let mut file = tokio::fs::File::create(&part_path).await?;

            use tokio::io::AsyncWriteExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(AssistantError::Http)?;
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;
                on_status(AssistantState::Downloading { downloaded, total });
            }
            file.flush().await?;
        }

        // ── SHA-256 verify ────────────────────────────────────────────────────
        info!("[assistant] verifying Ollama zip checksum");
        {
            let data = tokio::fs::read(&part_path).await?;
            let actual = hex::encode(Sha256::digest(&data));
            if actual != sha256_expected {
                let _ = tokio::fs::remove_file(&part_path).await;
                return Err(AssistantError::OllamaInstallFailed(format!(
                    "SHA-256 mismatch: expected {sha256_expected}, got {actual}"
                )));
            }
        }
        info!("[assistant] Ollama zip checksum ok");

        // ── Extract zip into bin_dir ──────────────────────────────────────────
        let part_path_clone = part_path.clone();
        let bin_dir_clone = bin_dir.clone();
        tokio::task::spawn_blocking(move || extract_zip(&part_path_clone, &bin_dir_clone))
            .await
            .map_err(|e| AssistantError::OllamaInstallFailed(e.to_string()))??;

        let _ = tokio::fs::remove_file(&part_path).await;

        // Mark executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let bin_name = "ollama";
            let bin = bin_dir.join(bin_name);
            if bin.exists() {
                let mut perms = std::fs::metadata(&bin)?.permissions();
                perms.set_mode(perms.mode() | 0o755);
                std::fs::set_permissions(&bin, perms)?;
            }
            // Also mark dylibs/so files executable (some platforms require it)
            for entry in std::fs::read_dir(&bin_dir)?.flatten() {
                let p = entry.path();
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext == "dylib" || ext == "so" {
                    if let Ok(mut perms) = std::fs::metadata(&p).map(|m| m.permissions()) {
                        perms.set_mode(perms.mode() | 0o755);
                        let _ = std::fs::set_permissions(&p, perms);
                    }
                }
            }
        }

        // Write version marker
        let marker = serde_json::json!({ "version": OLLAMA_VERSION });
        tokio::fs::write(
            self.config.data_dir.join(VERSION_FILE),
            serde_json::to_string_pretty(&marker)?,
        )
        .await?;

        info!(
            "[assistant] Ollama {} provisioned at {}",
            OLLAMA_VERSION,
            bin_dir.display()
        );
        self.engine_managed_install = true;
        Ok(())
    }
}

// ── AssistantBackend impl ──────────────────────────────────────────────────────

impl AssistantBackend for OllamaBackend {
    fn probe(&mut self) -> AssistantState {
        let binary_present = self.ollama_binary_present();
        let private_install = self.private_install_present();
        info!("[assistant] probe: binary_present={} private_install={} data_dir={}", binary_present, private_install, self.config.data_dir.display());
        if binary_present {
            self.state = AssistantState::ServerDown;
        } else {
            self.state = AssistantState::NeedsInstall;
        }
        if private_install {
            self.engine_managed_install = true;
        }
        info!("[assistant] probe: state={:?} engine_managed_install={}", self.state, self.engine_managed_install);
        self.state.clone()
    }

    async fn ensure_ready(
        &mut self,
        on_status: impl Fn(AssistantState) + Send + Sync + 'static,
    ) -> Result<()> {
        on_status(AssistantState::NotInstalled);

        // Step 1: provision if neither private nor system Ollama is present
        if !self.ollama_binary_present() {
            on_status(AssistantState::NeedsInstall);
            self.provision_ollama(&on_status).await?;
        } else if self.private_install_present() {
            self.engine_managed_install = true;
        }

        // Step 2: start the server
        self.ensure_server_running().await?;
        on_status(AssistantState::ServerDown);

        // Step 3: pull the model
        self.pull_model(on_status).await
    }

    async fn pull_model(
        &mut self,
        on_status: impl Fn(AssistantState) + Send + Sync + 'static,
    ) -> Result<()> {
        self.ensure_server_running().await?;

        let model_id = self.config.model_id.clone();
        if self.model_present(&model_id).await {
            info!("[assistant] model {} already present", model_id);
            self.state = AssistantState::Ready;
            on_status(AssistantState::Ready);
            return Ok(());
        }

        info!("[assistant] pulling model {}", model_id);
        on_status(AssistantState::Downloading { downloaded: 0, total: None });

        let resp = self
            .client
            .post(format!("{}/api/pull", OLLAMA_BASE))
            .json(&serde_json::json!({ "name": model_id, "stream": true }))
            .send()
            .await
            .map_err(AssistantError::Http)?;

        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(AssistantError::Http)?;
            for line in chunk.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(line) {
                    let completed = val.get("completed").and_then(|v| v.as_u64());
                    let total = val.get("total").and_then(|v| v.as_u64());
                    if let Some(downloaded) = completed {
                        on_status(AssistantState::Downloading { downloaded, total });
                    }
                    if val.get("status").and_then(|v| v.as_str()) == Some("success") {
                        break;
                    }
                }
            }
        }

        self.state = AssistantState::Ready;
        on_status(AssistantState::Ready);
        info!("[assistant] model {} ready", model_id);

        // Also pull the embedding model if it differs from the chat model.
        let embed_id = self.config.embedding_model.clone();
        if embed_id != model_id {
            if self.model_present(&embed_id).await {
                info!("[assistant] embedding model {} already present", embed_id);
            } else {
                info!("[assistant] pulling embedding model {}", embed_id);
                on_status(AssistantState::Downloading { downloaded: 0, total: None });
                let resp = self
                    .client
                    .post(format!("{}/api/pull", OLLAMA_BASE))
                    .json(&serde_json::json!({ "name": embed_id, "stream": true }))
                    .send()
                    .await
                    .map_err(AssistantError::Http)?;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(AssistantError::Http)?;
                    for line in chunk.split(|&b| b == b'\n') {
                        if line.is_empty() { continue; }
                        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(line) {
                            let completed = val.get("completed").and_then(|v| v.as_u64());
                            let total = val.get("total").and_then(|v| v.as_u64());
                            if let Some(downloaded) = completed {
                                on_status(AssistantState::Downloading { downloaded, total });
                            }
                            if val.get("status").and_then(|v| v.as_str()) == Some("success") {
                                break;
                            }
                        }
                    }
                }
                info!("[assistant] embedding model {} ready", embed_id);
            }
        }

        Ok(())
    }

    async fn chat(
        &mut self,
        messages: Vec<ChatMessage>,
        token_tx: Sender<ChatToken>,
        cancel_rx: tokio::sync::oneshot::Receiver<()>,
    ) -> Result<()> {
        info!("[assistant] chat: ensuring server running");
        self.ensure_server_running().await?;
        info!("[assistant] chat: sending {} messages to model {}", messages.len(), self.config.model_id);

        let payload = serde_json::json!({
            "model": self.config.model_id,
            "messages": messages,
            "stream": true,
            "options": {
                "num_ctx": self.config.context_window,
                "keep_alive": "10m",
            }
        });

        let resp = self
            .client
            .post(format!("{}/api/chat", OLLAMA_BASE))
            .json(&payload)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(AssistantError::Http)?;

        let mut stream = resp.bytes_stream();
        let mut cancel_rx = cancel_rx;

        loop {
            tokio::select! {
                _ = &mut cancel_rx => {
                    let _ = token_tx.send(ChatToken { text: String::new(), done: true });
                    return Ok(());
                }
                chunk = stream.next() => {
                    let Some(chunk) = chunk else { break; };
                    let chunk = chunk.map_err(AssistantError::Http)?;
                    for line in chunk.split(|&b| b == b'\n') {
                        if line.is_empty() { continue; }
                        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(line) {
                            let done = val.get("done").and_then(|v| v.as_bool()).unwrap_or(false);
                            let text = val
                                .get("message")
                                .and_then(|m| m.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let _ = token_tx.send(ChatToken { text, done });
                            if done { return Ok(()); }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let payload = serde_json::json!({
            "model": self.config.embedding_model,
            "prompt": text,
        });
        let resp = self
            .client
            .post(format!("{}/api/embeddings", OLLAMA_BASE))
            .json(&payload)
            .send()
            .await
            .map_err(AssistantError::Http)?;
        let body: serde_json::Value = resp.json().await.map_err(AssistantError::Http)?;
        let embedding = body
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| AssistantError::Internal("missing embedding in response".into()))?;
        Ok(embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect())
    }

    async fn is_server_alive(&self) -> bool {
        self.server_alive().await
    }

    fn status(&self) -> AssistantState {
        self.state.clone()
    }

    fn detect_ram_bytes(&self) -> Option<u64> {
        detect_system_ram()
    }
}

impl crate::rag::AsyncEmbedFn for OllamaBackend {
    async fn embed(&self, text: &str) -> crate::Result<Vec<f32>> {
        AssistantBackend::embed(self, text).await
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Returns `(asset_filename, expected_sha256)` for the current platform/arch.
fn platform_asset() -> Result<(String, &'static str)> {
    #[cfg(target_os = "macos")]
    {
        return Ok((
            format!("ollama-{}-macos-universal.zip", ASSET_RELEASE_TAG),
            OLLAMA_SHA256_MACOS_UNIVERSAL,
        ));
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Ok((
            format!("ollama-{}-linux-x86_64.zip", ASSET_RELEASE_TAG),
            OLLAMA_SHA256_LINUX_X86_64,
        ));
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Ok((
            format!("ollama-{}-linux-aarch64.zip", ASSET_RELEASE_TAG),
            OLLAMA_SHA256_LINUX_AARCH64,
        ));
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Ok((
            format!("ollama-{}-windows-x86_64.zip", ASSET_RELEASE_TAG),
            OLLAMA_SHA256_WINDOWS_X86_64,
        ));
    }
    #[allow(unreachable_code)]
    Err(AssistantError::OllamaInstallFailed(
        "Automatic Ollama install is not supported on this platform. \
         Please install Ollama manually from https://ollama.com"
            .into(),
    ))
}

/// Extract all entries from a zip into `dest_dir`, preserving subdirectory
/// structure.  This is used both for platform zips (flat or with lib/ subdir)
/// and any future layout.
fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(zip_path)
        .map_err(|e| AssistantError::OllamaInstallFailed(format!("open zip: {e}")))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| AssistantError::OllamaInstallFailed(format!("read zip: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| AssistantError::OllamaInstallFailed(format!("zip entry {i}: {e}")))?;

        if entry.is_dir() {
            continue;
        }

        // Normalise path separators and strip any leading "." or absolute prefix
        let entry_name = entry.name().replace('\\', "/");
        let entry_name = entry_name.trim_start_matches("./").trim_start_matches('/');

        let out_path = dest_dir.join(entry_name);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AssistantError::OllamaInstallFailed(format!("mkdir {}: {e}", parent.display())))?;
        }

        let mut out = std::fs::File::create(&out_path)
            .map_err(|e| AssistantError::OllamaInstallFailed(format!("create {}: {e}", out_path.display())))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| AssistantError::OllamaInstallFailed(format!("extract {entry_name}: {e}")))?;
    }
    Ok(())
}

/// Prepend `dir` to the PATH environment variable for a spawned command.
fn prepend_path_env(cmd: &mut tokio::process::Command, dir: &Path) {
    let current = std::env::var("PATH").unwrap_or_default();
    let sep = if cfg!(target_os = "windows") { ";" } else { ":" };
    cmd.env("PATH", format!("{}{sep}{current}", dir.display()));
}

fn detect_system_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let data = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in data.lines() {
            if line.starts_with("MemTotal:") {
                let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(target_os = "windows")]
    {
        None
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}
