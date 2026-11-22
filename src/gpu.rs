//! GPU acceleration for the Ollama install Corylus manages.
//!
//! The base archive is CPU-only. When a usable graphics card is found the
//! writer can download one add-on (CUDA or Vulkan) that holds Ollama's
//! `lib/ollama/<backend>/` directory. The backend is then chosen with
//! `OLLAMA_LLM_LIBRARY`, checked against the server log after every restart,
//! and dropped (CUDA → Vulkan → CPU) when it does not come up.
//!
//! State lives in `<data_dir>/gpu.json`. No file means GPU support was never
//! installed, and the server is started exactly as before.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{error::AssistantError, Result};

pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_AMD: u32 = 0x1002;
pub const VENDOR_INTEL: u32 = 0x8086;

const GIB: u64 = 1024 * 1024 * 1024;
/// Below this a card is an integrated GPU or too small to beat the CPU.
const MIN_VRAM_BYTES: u64 = 2 * GIB;
/// CUDA 13 (Ollama's `cuda_v13`) needs an R580 or newer driver.
const CUDA_MIN_DRIVER_MAJOR: u32 = 580;

const STATE_FILE: &str = "gpu.json";
/// GitHub release holding the add-ons. The CPU base stays on `v0.30.10`.
const ADDON_RELEASE_TAG: &str = "v0.30.10-gpu";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cuda,
    Vulkan,
    Cpu,
}

impl Backend {
    /// Value for `OLLAMA_LLM_LIBRARY` (the directory name under `lib/ollama/`).
    pub fn llm_library(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda_v13",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }

    /// Matches the `library=` value of Ollama's `inference compute` log line.
    fn matches_log_library(self, library: &str) -> bool {
        library.eq_ignore_ascii_case(match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        })
    }
}

/// A graphics adapter as the OS reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct Adapter {
    pub vendor_id: u32,
    pub name: String,
    /// Dedicated video memory (0 when the OS does not say).
    pub vram_bytes: u64,
    /// NVIDIA driver version as (major, minor), e.g. (610, 88).
    pub nvidia_driver: Option<(u32, u32)>,
    pub software: bool,
}

pub fn vendor_name(vendor_id: u32) -> Option<&'static str> {
    match vendor_id {
        VENDOR_NVIDIA => Some("NVIDIA"),
        VENDOR_AMD => Some("AMD"),
        VENDOR_INTEL => Some("Intel"),
        _ => None,
    }
}

/// True when the adapter is a real NVIDIA/AMD/Intel GPU with room for the model.
pub fn usable(a: &Adapter, model_bytes: u64) -> bool {
    !a.software
        && vendor_name(a.vendor_id).is_some()
        && !a.name.contains("Microsoft Basic Render")
        && a.vram_bytes >= MIN_VRAM_BYTES
        && a.vram_bytes as f64 >= model_bytes as f64 * 1.2
}

/// The usable adapter with the most VRAM.
pub fn best_adapter(adapters: &[Adapter], model_bytes: u64) -> Option<&Adapter> {
    adapters.iter().filter(|a| usable(a, model_bytes)).max_by_key(|a| a.vram_bytes)
}

/// Pick the backend for `adapter`, skipping backends that already failed.
pub fn choose(adapter: Option<&Adapter>, failed: &[Backend]) -> Backend {
    let Some(a) = adapter else { return Backend::Cpu };
    let cuda_driver_ok = a.nvidia_driver.is_some_and(|(major, _)| major >= CUDA_MIN_DRIVER_MAJOR);
    if a.vendor_id == VENDOR_NVIDIA && cuda_driver_ok && !failed.contains(&Backend::Cuda) {
        Backend::Cuda
    } else if !failed.contains(&Backend::Vulkan) {
        Backend::Vulkan
    } else {
        Backend::Cpu
    }
}

/// NVIDIA driver version from the DXGI user-mode driver version,
/// e.g. 32.0.16.1088 → (610, 88): the last digit of the third part and the
/// fourth part read as "6" + "10.88".
pub fn nvidia_driver_from_umd(umd: u64) -> (u32, u32) {
    let c = ((umd >> 16) & 0xFFFF) as u32;
    let d = (umd & 0xFFFF) as u32;
    ((c % 10) * 100 + d / 100, d % 100)
}

// ── Add-on archives ────────────────────────────────────────────────────────────

pub struct Addon {
    pub backend: Backend,
    pub file: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// Pinned add-ons produced by `scripts/prepare_ollama.py`. Platforms without
/// an entry stay on the CPU (Linux is not a shipped target; macOS uses Metal).
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub const ADDONS: &[Addon] = &[
    Addon {
        backend: Backend::Cuda,
        file: "ollama-v0.30.10-windows-x86_64-cuda.zip",
        sha256: "d3ca5af7ac454ded26954b253a778e22cc50df5c5810996dddcafe1a9ae8e8f1",
        size: 485_921_519,
    },
    Addon {
        backend: Backend::Vulkan,
        file: "ollama-v0.30.10-windows-x86_64-vulkan.zip",
        sha256: "d6ee89b2d7745444a23ef88df593b695eaf0360f2c45062601cb1c90d0d500a5",
        size: 23_691_910,
    },
];
#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub const ADDONS: &[Addon] = &[];

pub fn addon(backend: Backend) -> Option<&'static Addon> {
    ADDONS.iter().find(|a| a.backend == backend)
}

pub fn addon_url(a: &Addon) -> String {
    format!("{}/{}/{}", crate::ollama::ASSET_BASE_URL, ADDON_RELEASE_TAG, a.file)
}

// ── Persistent state ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Failure {
    pub backend: Backend,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuState {
    #[serde(default)]
    pub installed: Vec<Backend>,
    /// Backend the server is started with (unless `cpu_only`).
    #[serde(default)]
    pub active: Option<Backend>,
    /// The writer chose the processor.
    #[serde(default)]
    pub cpu_only: bool,
    #[serde(default)]
    pub failed: Vec<Failure>,
    /// Why the last start did not use the selected backend.
    #[serde(default)]
    pub note: Option<String>,
}

impl GpuState {
    pub fn effective(&self) -> Backend {
        if self.cpu_only { Backend::Cpu } else { self.active.unwrap_or(Backend::Cpu) }
    }

    pub fn failed_backends(&self) -> Vec<Backend> {
        self.failed.iter().map(|f| f.backend).collect()
    }

    /// Record that `wanted` did not come up (Ollama reported `observed`) and
    /// return the backend to try next, or `None` when there is nothing left.
    pub fn record_failure(&mut self, wanted: Backend, observed: &[String]) -> Option<Backend> {
        let seen = if observed.is_empty() { "nothing".to_string() } else { observed.join(",") };
        let reason = format!(
            "{} did not start: Ollama reported library={seen}",
            wanted.llm_library()
        );
        self.note = Some(reason.clone());
        if wanted == Backend::Cpu {
            return None;
        }
        self.failed.retain(|f| f.backend != wanted);
        self.failed.push(Failure { backend: wanted, reason });
        let next = match wanted {
            Backend::Cuda if self.installed.contains(&Backend::Vulkan) => Backend::Vulkan,
            _ => Backend::Cpu,
        };
        self.active = Some(next);
        Some(next)
    }
}

pub fn state_path(data_dir: &Path) -> PathBuf {
    data_dir.join(STATE_FILE)
}

pub fn load_state(data_dir: &Path) -> Option<GpuState> {
    let data = std::fs::read_to_string(state_path(data_dir)).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn save_state(data_dir: &Path, state: &GpuState) -> Result<()> {
    let path = state_path(data_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

// ── Log parsing ────────────────────────────────────────────────────────────────

/// The `library=` values of every `inference compute` line in `log`,
/// e.g. `["CUDA"]` or `["cpu"]`.
pub fn inference_libraries(log: &str) -> Vec<String> {
    log.lines()
        .filter(|l| l.contains("msg=\"inference compute\""))
        .filter_map(|l| {
            let rest = &l[l.find(" library=")? + " library=".len()..];
            Some(rest.split_whitespace().next()?.trim_matches('"').to_string())
        })
        .collect()
}

pub fn libraries_include(observed: &[String], backend: Backend) -> bool {
    observed.iter().any(|l| backend.matches_log_library(l))
}

// ── Install ────────────────────────────────────────────────────────────────────

fn lib_dir(bin_dir: &Path) -> PathBuf {
    bin_dir.join("lib").join("ollama")
}

/// Rename with a few retries: a just-killed process can hold a DLL for a moment.
fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut last = None;
    for _ in 0..10 {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => last = Some(e),
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(last.unwrap())
}

/// Move a flat install (`bin/ollama.exe` beside its DLLs) to Ollama's layout
/// (`bin/lib/ollama/*`), so `lib/ollama/<backend>/` is found. Idempotent; the
/// managed server must be stopped.
pub fn migrate_to_upstream_layout(bin_dir: &Path) -> std::io::Result<()> {
    let lib = lib_dir(bin_dir);
    std::fs::create_dir_all(&lib)?;
    for entry in std::fs::read_dir(bin_dir)?.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if !path.is_file() || name == "ollama.exe" || name == "ollama" {
            continue;
        }
        rename_retry(&path, &lib.join(&name))?;
    }
    Ok(())
}

/// Stream an add-on to `<data_dir>/<file>.part`, capped at its pinned size and
/// checked against its pinned SHA-256. Any failure removes the part file.
pub async fn download_addon(
    a: &Addon,
    data_dir: &Path,
    cancel: &AtomicBool,
    on_progress: impl Fn(u64, u64),
) -> Result<PathBuf> {
    let part = data_dir.join(format!("{}.part", a.file));
    let result = download_to(a, &part, cancel, on_progress).await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&part).await;
    }
    result.map(|_| part)
}

async fn download_to(
    a: &Addon,
    part: &Path,
    cancel: &AtomicBool,
    on_progress: impl Fn(u64, u64),
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()?;
    let resp = client.get(addon_url(a)).send().await?.error_for_status()?;
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(part).await?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Err(AssistantError::Cancelled);
        }
        let chunk = chunk?;
        downloaded += chunk.len() as u64;
        if downloaded > a.size {
            return Err(AssistantError::OllamaInstallFailed(format!(
                "{} is larger than the expected {} bytes",
                a.file, a.size
            )));
        }
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        on_progress(downloaded, a.size);
    }
    file.flush().await?;
    file.sync_all().await?;
    let actual = hex::encode(hasher.finalize());
    if downloaded != a.size || actual != a.sha256 {
        return Err(AssistantError::OllamaInstallFailed(format!(
            "{}: expected {} bytes sha256 {}, got {downloaded} bytes sha256 {actual}",
            a.file, a.size, a.sha256
        )));
    }
    Ok(())
}

/// Migrate the layout, extract the add-on into a staging directory and rename
/// it into `bin/lib/ollama/<backend>/`. The managed server must be stopped.
pub fn install_addon_zip(zip: &Path, bin_dir: &Path, backend: Backend) -> Result<()> {
    migrate_to_upstream_layout(bin_dir)?;
    let staging = bin_dir.join("lib").join(format!(".staging-{}", backend.llm_library()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    let result = (|| {
        crate::ollama::extract_zip(zip, &staging)?;
        let src = lib_dir(&staging).join(backend.llm_library());
        if !src.is_dir() {
            return Err(AssistantError::OllamaInstallFailed(format!(
                "add-on has no lib/ollama/{}/",
                backend.llm_library()
            )));
        }
        let dst = lib_dir(bin_dir).join(backend.llm_library());
        if dst.exists() {
            std::fs::remove_dir_all(&dst)?;
        }
        rename_retry(&src, &dst)?;
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&staging);
    result
}

// ── Detection ──────────────────────────────────────────────────────────────────

/// Graphics adapters on this machine. Empty on macOS (Ollama uses Metal there).
pub fn detect_adapters() -> Vec<Adapter> {
    #[cfg(target_os = "windows")]
    {
        detect_dxgi()
    }
    #[cfg(target_os = "linux")]
    {
        detect_sysfs(Path::new("/sys/class/drm"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Vec::new()
    }
}

/// Linux: `/sys/class/drm/cardN/device/{vendor,mem_info_vram_total}`.
/// shortcut: only amdgpu reports VRAM here, so NVIDIA/Intel read as 0 (not usable);
/// upgrade when Linux becomes a shipped target with add-ons.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub fn detect_sysfs(drm: &Path) -> Vec<Adapter> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(drm) else { return out };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.strip_prefix("card").is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())) {
            continue;
        }
        let dev = entry.path().join("device");
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).ok().map(|s| s.trim().to_string());
        let Some(vendor_id) = read("vendor").and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()) else { continue };
        let vram_bytes = read("mem_info_vram_total").and_then(|v| v.parse().ok()).unwrap_or(0);
        out.push(Adapter {
            vendor_id,
            name: format!("{} GPU", vendor_name(vendor_id).unwrap_or("Unknown")),
            vram_bytes,
            nvidia_driver: None,
            software: false,
        });
    }
    out
}

/// Windows: enumerate DXGI adapters. Declared by hand, like `GlobalMemoryStatusEx`
/// in `ollama.rs`, to avoid a windows-sys dependency.
#[cfg(target_os = "windows")]
fn detect_dxgi() -> Vec<Adapter> {
    use std::ffi::c_void;

    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);
    #[repr(C)]
    struct AdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        luid_low: u32,
        luid_high: i32,
        flags: u32,
    }
    const IID_IDXGI_FACTORY1: Guid =
        Guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);
    const IID_IDXGI_DEVICE: Guid =
        Guid(0x54ec77fa, 0x1377, 0x44e6, [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c]);
    const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;

    #[link(name = "dxgi")]
    extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> i32;
    }
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    type EnumAdapters1 = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
    type CheckInterfaceSupport = unsafe extern "system" fn(*mut c_void, *const Guid, *mut i64) -> i32;
    type GetDesc1 = unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32;

    /// Function pointer at `index` in a COM object's vtable.
    unsafe fn method<T: Copy>(obj: *mut c_void, index: usize) -> T {
        let vtbl = *(obj as *const *const usize);
        std::mem::transmute_copy(&*vtbl.add(index))
    }

    let mut out = Vec::new();
    // SAFETY: standard DXGI COM calls; vtable slots per dxgi.h (IUnknown 0–2,
    // IDXGIObject 3–6, IDXGIFactory 7–11, EnumAdapters1 = 12; IDXGIAdapter
    // CheckInterfaceSupport = 9, IDXGIAdapter1 GetDesc1 = 10). Every object we
    // get is released.
    unsafe {
        let mut factory: *mut c_void = std::ptr::null_mut();
        if CreateDXGIFactory1(&IID_IDXGI_FACTORY1, &mut factory) < 0 || factory.is_null() {
            return out;
        }
        let enum_adapters: EnumAdapters1 = method(factory, 12);
        let mut i = 0;
        loop {
            let mut adapter: *mut c_void = std::ptr::null_mut();
            if enum_adapters(factory, i, &mut adapter) < 0 || adapter.is_null() {
                break; // DXGI_ERROR_NOT_FOUND ends the list
            }
            i += 1;
            let mut desc: AdapterDesc1 = std::mem::zeroed();
            let get_desc: GetDesc1 = method(adapter, 10);
            if get_desc(adapter, &mut desc) >= 0 {
                let len = desc.description.iter().position(|&c| c == 0).unwrap_or(128);
                let mut umd: i64 = 0;
                let check: CheckInterfaceSupport = method(adapter, 9);
                let nvidia_driver = (desc.vendor_id == VENDOR_NVIDIA
                    && check(adapter, &IID_IDXGI_DEVICE, &mut umd) >= 0)
                    .then(|| nvidia_driver_from_umd(umd as u64));
                out.push(Adapter {
                    vendor_id: desc.vendor_id,
                    name: String::from_utf16_lossy(&desc.description[..len]).trim().to_string(),
                    vram_bytes: desc.dedicated_video_memory as u64,
                    nvidia_driver,
                    software: desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0,
                });
            }
            let release: Release = method(adapter, 2);
            release(adapter);
        }
        let release: Release = method(factory, 2);
        release(factory);
    }
    out
}

// ── Status for the UI ──────────────────────────────────────────────────────────

/// What Settings shows: the card, what runs now, and what could be installed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuStatus {
    /// False on macOS, without a Corylus-managed install, or without add-ons.
    pub supported: bool,
    pub device: Option<String>,
    pub vram_bytes: Option<u64>,
    pub active: Backend,
    pub recommended: Backend,
    /// Size of the add-on `recommended` needs; 0 when it is installed or CPU.
    pub download_bytes: u64,
    pub cpu_only: bool,
    pub note: Option<String>,
}

pub fn status(managed: bool, adapters: &[Adapter], model_bytes: u64, state: Option<&GpuState>) -> GpuStatus {
    let best = best_adapter(adapters, model_bytes);
    let failed = state.map(|s| s.failed_backends()).unwrap_or_default();
    let recommended = choose(best, &failed);
    let installed = state.is_some_and(|s| s.installed.contains(&recommended));
    let download_bytes = match addon(recommended) {
        Some(a) if !installed => a.size,
        _ => 0,
    };
    GpuStatus {
        supported: managed && !ADDONS.is_empty(),
        device: best.map(|a| a.name.clone()),
        vram_bytes: best.map(|a| a.vram_bytes),
        active: state.map(|s| s.effective()).unwrap_or(Backend::Cpu),
        recommended,
        download_bytes,
        cpu_only: state.is_some_and(|s| s.cpu_only),
        note: state.and_then(|s| s.note.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(vendor_id: u32, name: &str, vram_gb: u64, driver: Option<(u32, u32)>) -> Adapter {
        Adapter { vendor_id, name: name.into(), vram_bytes: vram_gb * GIB, nvidia_driver: driver, software: false }
    }

    const MODEL_1_7B: u64 = 1_400_000_000;

    #[test]
    fn vendor_ids_map_to_names() {
        assert_eq!(vendor_name(0x10DE), Some("NVIDIA"));
        assert_eq!(vendor_name(0x1002), Some("AMD"));
        assert_eq!(vendor_name(0x8086), Some("Intel"));
        assert_eq!(vendor_name(0x1414), None); // Microsoft
    }

    #[test]
    fn software_and_basic_render_adapters_are_not_usable() {
        let mut sw = card(VENDOR_NVIDIA, "NVIDIA GeForce RTX 4070", 12, Some((610, 88)));
        sw.software = true;
        let basic = card(0x1414, "Microsoft Basic Render Driver", 0, None);
        assert!(best_adapter(&[sw, basic], MODEL_1_7B).is_none());
    }

    #[test]
    fn integrated_gpu_with_little_dedicated_memory_is_not_usable() {
        let mut igpu = card(VENDOR_INTEL, "Intel(R) UHD Graphics 630", 0, None);
        igpu.vram_bytes = 128 * 1024 * 1024;
        assert!(!usable(&igpu, MODEL_1_7B));
    }

    #[test]
    fn best_adapter_is_the_one_with_most_vram() {
        let a = [card(VENDOR_INTEL, "Intel Arc A380", 6, None), card(VENDOR_NVIDIA, "RTX 4070", 12, Some((610, 88)))];
        assert_eq!(best_adapter(&a, MODEL_1_7B).unwrap().name, "RTX 4070");
    }

    #[test]
    fn recent_nvidia_driver_picks_cuda() {
        let c = card(VENDOR_NVIDIA, "RTX 4070", 12, Some((610, 88)));
        assert_eq!(choose(Some(&c), &[]), Backend::Cuda);
    }

    #[test]
    fn old_nvidia_driver_picks_vulkan() {
        let c = card(VENDOR_NVIDIA, "GTX 1080", 8, Some((552, 0)));
        assert_eq!(choose(Some(&c), &[]), Backend::Vulkan);
        let unknown = card(VENDOR_NVIDIA, "GTX 1080", 8, None);
        assert_eq!(choose(Some(&unknown), &[]), Backend::Vulkan);
    }

    #[test]
    fn amd_and_intel_pick_vulkan() {
        assert_eq!(choose(Some(&card(VENDOR_AMD, "RX 7800 XT", 16, None)), &[]), Backend::Vulkan);
        assert_eq!(choose(Some(&card(VENDOR_INTEL, "Arc A770", 16, None)), &[]), Backend::Vulkan);
    }

    #[test]
    fn no_card_or_card_too_small_picks_cpu() {
        assert_eq!(choose(None, &[]), Backend::Cpu);
        let small = [card(VENDOR_NVIDIA, "RTX 3050", 4, Some((610, 88)))];
        let fourteen_b = 9 * GIB;
        assert_eq!(choose(best_adapter(&small, fourteen_b), &[]), Backend::Cpu);
    }

    #[test]
    fn failed_backends_are_skipped() {
        let c = card(VENDOR_NVIDIA, "RTX 4070", 12, Some((610, 88)));
        assert_eq!(choose(Some(&c), &[Backend::Cuda]), Backend::Vulkan);
        assert_eq!(choose(Some(&c), &[Backend::Cuda, Backend::Vulkan]), Backend::Cpu);
    }

    #[test]
    fn nvidia_driver_is_read_from_the_umd_version() {
        // 32.0.16.1088 → 610.88
        let umd = (32u64 << 48) | (0 << 32) | (16 << 16) | 1088;
        assert_eq!(nvidia_driver_from_umd(umd), (610, 88));
        // 31.0.15.5222 → 552.22
        let umd = (31u64 << 48) | (15 << 16) | 5222;
        assert_eq!(nvidia_driver_from_umd(umd), (552, 22));
    }

    #[test]
    fn fallback_goes_cuda_then_vulkan_then_cpu() {
        let mut s = GpuState {
            installed: vec![Backend::Cuda, Backend::Vulkan],
            active: Some(Backend::Cuda),
            ..Default::default()
        };
        assert_eq!(s.record_failure(Backend::Cuda, &["cpu".into()]), Some(Backend::Vulkan));
        assert_eq!(s.active, Some(Backend::Vulkan));
        assert_eq!(s.note.as_deref(), Some("cuda_v13 did not start: Ollama reported library=cpu"));
        assert_eq!(s.record_failure(Backend::Vulkan, &[]), Some(Backend::Cpu));
        assert_eq!(s.failed_backends(), vec![Backend::Cuda, Backend::Vulkan]);
        assert_eq!(s.record_failure(Backend::Cpu, &[]), None);
        assert_eq!(s.effective(), Backend::Cpu);
    }

    #[test]
    fn cuda_failure_without_vulkan_installed_goes_to_cpu() {
        let mut s = GpuState { installed: vec![Backend::Cuda], active: Some(Backend::Cuda), ..Default::default() };
        assert_eq!(s.record_failure(Backend::Cuda, &["cpu".into()]), Some(Backend::Cpu));
        // Settings now offers Vulkan as the next download.
        let c = card(VENDOR_NVIDIA, "RTX 4070", 12, Some((610, 88)));
        let st = status(true, &[c], MODEL_1_7B, Some(&s));
        assert_eq!(st.recommended, Backend::Vulkan);
        assert_eq!(st.active, Backend::Cpu);
    }

    #[test]
    fn cpu_only_overrides_active() {
        let s = GpuState { active: Some(Backend::Cuda), cpu_only: true, ..Default::default() };
        assert_eq!(s.effective(), Backend::Cpu);
    }

    #[test]
    fn inference_compute_lines_are_parsed() {
        let log = r#"time=2026-11-22T12:23:19.793+01:00 level=INFO source=runner.go:60 msg="discovering available GPUs..."
time=2026-11-22T12:23:22.352+01:00 level=INFO source=types.go:32 msg="inference compute" id=0 filter_id=0 library=CUDA compute=8.9 name=CUDA0 description="NVIDIA GeForce RTX 4070" libdirs=ollama,cuda_v13 driver=13.3
time=2026-11-22T12:26:57.537+01:00 level=INFO source=types.go:50 msg="inference compute" id=cpu library=cpu compute="" name=cpu"#;
        let libs = inference_libraries(log);
        assert_eq!(libs, vec!["CUDA", "cpu"]);
        assert!(libraries_include(&libs, Backend::Cuda));
        assert!(!libraries_include(&libs, Backend::Vulkan));
        assert!(libraries_include(&["Vulkan".into()], Backend::Vulkan));
    }

    #[test]
    fn migration_moves_libraries_and_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path();
        for f in ["ollama.exe", "ggml-base.dll", "llama-server.exe"] {
            std::fs::write(bin.join(f), f).unwrap();
        }
        migrate_to_upstream_layout(bin).unwrap();
        migrate_to_upstream_layout(bin).unwrap();
        assert!(bin.join("ollama.exe").exists());
        assert!(!bin.join("ggml-base.dll").exists());
        assert!(bin.join("lib/ollama/ggml-base.dll").exists());
        assert!(bin.join("lib/ollama/llama-server.exe").exists());
    }

    #[test]
    fn addon_zip_installs_into_lib_ollama() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("ollama.exe"), "x").unwrap();
        std::fs::write(bin.join("ggml-base.dll"), "x").unwrap();
        let zip_path = tmp.path().join("addon.zip");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
        zw.start_file("lib/ollama/vulkan/ggml-vulkan.dll", zip::write::SimpleFileOptions::default()).unwrap();
        zw.write_all(b"vk").unwrap();
        zw.finish().unwrap();
        install_addon_zip(&zip_path, &bin, Backend::Vulkan).unwrap();
        assert_eq!(std::fs::read(bin.join("lib/ollama/vulkan/ggml-vulkan.dll")).unwrap(), b"vk");
        assert!(bin.join("lib/ollama/ggml-base.dll").exists());
        assert!(!bin.join("lib/.staging-vulkan").exists());
        // An archive without the expected directory is refused and leaves nothing behind.
        assert!(install_addon_zip(&zip_path, &bin, Backend::Cuda).is_err());
        assert!(!bin.join("lib/ollama/cuda_v13").exists());
    }

    #[test]
    fn sysfs_detection_reads_vendor_and_vram() {
        let tmp = tempfile::tempdir().unwrap();
        let card0 = tmp.path().join("card0/device");
        std::fs::create_dir_all(&card0).unwrap();
        std::fs::write(card0.join("vendor"), "0x1002\n").unwrap();
        std::fs::write(card0.join("mem_info_vram_total"), "17163091968\n").unwrap();
        std::fs::create_dir_all(tmp.path().join("card0-DP-1")).unwrap(); // connector, skipped
        let a = detect_sysfs(tmp.path());
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].vendor_id, VENDOR_AMD);
        assert_eq!(a[0].vram_bytes, 17_163_091_968);
    }

    /// Smoke test on the real machine: enumeration must not crash; prints what it found.
    #[test]
    fn detects_adapters_on_this_machine() {
        for a in detect_adapters() {
            println!("{:#06x} {} {} MiB driver={:?} software={}", a.vendor_id, a.name, a.vram_bytes >> 20, a.nvidia_driver, a.software);
        }
    }

    #[test]
    fn state_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_state(tmp.path()).is_none());
        let s = GpuState { installed: vec![Backend::Cuda], active: Some(Backend::Cuda), ..Default::default() };
        save_state(tmp.path(), &s).unwrap();
        assert_eq!(load_state(tmp.path()), Some(s));
    }
}
