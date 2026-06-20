//! Ollama HTTP backend implementation.
//!
//! Probes for an existing Ollama server, starts one if the binary is present,
//! or reports NeedsInstall. All inference traffic stays on 127.0.0.1:11434.

use std::sync::mpsc::Sender;
use std::time::Duration;
use tracing::{info, warn};

use crate::{
    AssistantState, EngineConfig, Result,
    backend::{AssistantBackend, ChatMessage, ChatToken},
    error::AssistantError,
};

const OLLAMA_BASE: &str = "http://127.0.0.1:11434";
const HEALTH_TIMEOUT_SECS: u64 = 3;

pub struct OllamaBackend {
    pub config: EngineConfig,
    pub state: AssistantState,
    client: reqwest::Client,
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
        }
    }

    async fn server_alive(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", OLLAMA_BASE))
            .timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    async fn ensure_server_running(&mut self) -> Result<()> {
        if self.server_alive().await {
            return Ok(());
        }
        // Try starting ollama serve
        if self.ollama_binary_present() {
            info!("[assistant] starting ollama serve");
            let _ = tokio::process::Command::new("ollama")
                .arg("serve")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
            // Wait up to 8 seconds for server to come up
            for _ in 0..16 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if self.server_alive().await {
                    info!("[assistant] ollama server started");
                    return Ok(());
                }
            }
            warn!("[assistant] ollama serve did not start in time");
            return Err(AssistantError::OllamaServerDown);
        }
        Err(AssistantError::OllamaNotInstalled)
    }

    fn ollama_binary_present(&self) -> bool {
        which::which("ollama").is_ok()
    }

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
                        .map(|n| n == model_id || n.starts_with(&format!("{}:", model_id.split(':').next().unwrap_or(model_id))))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }
}

impl AssistantBackend for OllamaBackend {
    fn probe(&mut self) -> AssistantState {
        // Synchronous probe — used at startup to set initial state without blocking the UI.
        // A running server is the most optimistic outcome; no server + no binary = needs_install.
        if self.ollama_binary_present() {
            self.state = AssistantState::ServerDown;
        } else {
            self.state = AssistantState::NeedsInstall;
        }
        self.state.clone()
    }

    async fn ensure_ready(
        &mut self,
        on_status: impl Fn(AssistantState) + Send + 'static,
    ) -> Result<()> {
        on_status(AssistantState::NotInstalled);

        // Step 1: install Ollama if needed (consent already given by caller)
        if !self.ollama_binary_present() {
            on_status(AssistantState::NeedsInstall);
            self.install_ollama().await?;
        }

        // Step 2: start the server
        self.ensure_server_running().await?;
        on_status(AssistantState::ServerDown);

        // Step 3: pull the model
        self.pull_model(on_status).await
    }

    async fn pull_model(
        &mut self,
        on_status: impl Fn(AssistantState) + Send + 'static,
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

        use futures_util::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut total_downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(AssistantError::Http)?;
            // Ollama pull stream: newline-delimited JSON with {"status","completed","total"}
            for line in chunk.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(line) {
                    let completed = val.get("completed").and_then(|v| v.as_u64());
                    let total = val.get("total").and_then(|v| v.as_u64());
                    if let Some(c) = completed {
                        total_downloaded = c;
                        on_status(AssistantState::Downloading {
                            downloaded: total_downloaded,
                            total,
                        });
                    }
                    let status = val.get("status").and_then(|v| v.as_str()).unwrap_or("");
                    if status == "success" {
                        break;
                    }
                }
            }
        }

        self.state = AssistantState::Ready;
        on_status(AssistantState::Ready);
        info!("[assistant] model {} ready", model_id);
        Ok(())
    }

    async fn chat(
        &mut self,
        messages: Vec<ChatMessage>,
        token_tx: Sender<ChatToken>,
        cancel_rx: tokio::sync::oneshot::Receiver<()>,
    ) -> Result<()> {
        self.ensure_server_running().await?;

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

        use futures_util::StreamExt;
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

    fn status(&self) -> AssistantState {
        self.state.clone()
    }

    fn detect_ram_bytes(&self) -> Option<u64> {
        detect_system_ram()
    }
}

impl OllamaBackend {
    async fn install_ollama(&mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            self.install_ollama_macos().await
        }
        #[cfg(target_os = "linux")]
        {
            self.install_ollama_linux().await
        }
        #[cfg(target_os = "windows")]
        {
            self.install_ollama_windows().await
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            Err(AssistantError::OllamaInstallFailed(
                "Automatic Ollama install is not supported on this platform. \
                 Please install Ollama manually from https://ollama.com"
                    .into(),
            ))
        }
    }

    #[cfg(target_os = "linux")]
    async fn install_ollama_linux(&mut self) -> Result<()> {
        info!("[assistant] installing Ollama on Linux via install.sh");
        let output = tokio::process::Command::new("sh")
            .args(["-c", "curl -fsSL https://ollama.com/install.sh | sh"])
            .output()
            .await
            .map_err(|e| AssistantError::OllamaInstallFailed(e.to_string()))?;
        if !output.status.success() {
            return Err(AssistantError::OllamaInstallFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn install_ollama_macos(&mut self) -> Result<()> {
        info!("[assistant] installing Ollama on macOS via install.sh");
        let output = tokio::process::Command::new("sh")
            .args(["-c", "curl -fsSL https://ollama.com/install.sh | sh"])
            .output()
            .await
            .map_err(|e| AssistantError::OllamaInstallFailed(e.to_string()))?;
        if !output.status.success() {
            return Err(AssistantError::OllamaInstallFailed(format!(
                "Ollama install failed. Please install manually from https://ollama.com\n{}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    async fn install_ollama_windows(&mut self) -> Result<()> {
        info!("[assistant] installing Ollama on Windows");
        // Download OllamaSetup.exe to a temp path and run it (per-user, no admin required)
        let tmp = std::env::temp_dir().join("OllamaSetup.exe");
        let resp = reqwest::get("https://ollama.com/download/OllamaSetup.exe")
            .await
            .map_err(AssistantError::Http)?;
        let bytes = resp.bytes().await.map_err(AssistantError::Http)?;
        tokio::fs::write(&tmp, &bytes).await?;
        let output = tokio::process::Command::new(&tmp)
            .args(["/SILENT"])
            .output()
            .await
            .map_err(|e| AssistantError::OllamaInstallFailed(e.to_string()))?;
        if !output.status.success() {
            return Err(AssistantError::OllamaInstallFailed(format!(
                "OllamaSetup.exe failed. Please install manually from https://ollama.com\n{}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(())
    }
}

fn detect_system_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let data = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in data.lines() {
            if line.starts_with("MemTotal:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)?
                    .parse()
                    .ok()?;
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
        // Use GlobalMemoryStatusEx via winapi — fall back to None for now
        None
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}
