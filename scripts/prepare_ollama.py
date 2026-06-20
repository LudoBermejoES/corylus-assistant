#!/usr/bin/env python3
"""
Prepare vendored Ollama platform zips for the rust-assistant crate.

For each supported platform this script:
  1. Fetches the latest (or a pinned) Ollama release from GitHub.
  2. Downloads the upstream archive.
  3. Extracts ONLY the files Corylus needs:
       macOS  — ollama binary + all *.dylib files
       Linux  — bin/ollama + lib/ollama/*.so (CPU libs only; GPU/ROCm excluded)
       Windows — ollama.exe + all *.dll files
  4. Repacks them into a minimal zip named
       ollama-{version}-{platform}-{arch}.zip
  5. Prints the SHA-256 of each output zip so you can paste them into
     ollama.rs.

Output files are written to  scripts/dist/  and are meant to be uploaded
to a GitHub Release of the rust-assistant repo (or wherever asset_base_url
points) and committed nowhere — only the SHA256s live in source.

Usage
-----
  # latest release
  python3 scripts/prepare_ollama.py

  # pin a specific version
  python3 scripts/prepare_ollama.py --version v0.30.10

  # build only one platform (for CI on that OS)
  python3 scripts/prepare_ollama.py --platform macos

Requirements: Python 3.9+, no extra packages (uses stdlib only).
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GITHUB_RELEASE_API = "https://api.github.com/repos/ollama/ollama/releases"

# Upstream asset names for each (platform, arch) pair.
# platform_key → (upstream_asset_name, arch_label)
UPSTREAM_ASSETS: dict[str, list[tuple[str, str]]] = {
    "macos": [
        # Universal binary — works on both Intel and Apple Silicon
        ("ollama-darwin.tgz", "universal"),
    ],
    "linux": [
        ("ollama-linux-amd64.tar.zst", "x86_64"),
        ("ollama-linux-arm64.tar.zst", "aarch64"),
    ],
    "windows": [
        ("ollama-windows-amd64.zip", "x86_64"),
    ],
}

# GPU / ROCm library prefixes that we strip from Linux to keep the zip small.
# Writers do not need CUDA; they can always fall back to CPU.
LINUX_GPU_PREFIXES = (
    "lib/ollama/cuda_v",
    "lib/ollama/rocm",
    "lib/ollama/oneapi",
)


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "corylus-prepare-ollama/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def resolve_version(pinned: str | None) -> tuple[str, list[dict]]:
    """Return (tag_name, assets_list) for the requested or latest release."""
    if pinned:
        data = fetch_json(f"{GITHUB_RELEASE_API}/tags/{pinned}")
    else:
        data = fetch_json(f"{GITHUB_RELEASE_API}/latest")
    return data["tag_name"], data["assets"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dest: Path) -> None:
    print(f"  Downloading {url.split('/')[-1]} …", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "corylus-prepare-ollama/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  Downloading {dest.name} … {pct:3d}% ({downloaded // 1_048_576} MB / {total // 1_048_576} MB)", end="", flush=True)
    print(f"\r  Downloaded  {dest.name} ({downloaded // 1_048_576} MB)          ")


# ── macOS ──────────────────────────────────────────────────────────────────────

def repack_macos(src: Path, out_zip: Path) -> None:
    """
    ollama-darwin.tgz contains ollama binary + *.dylib files (top-level and in
    mlx_metal_v3/ and mlx_metal_v4/ subdirs for Apple Silicon GPU acceleration).

    The tgz uses hard-links to produce versioned aliases
    (e.g. libggml-base.0.dylib → libggml-base.0.15.1.dylib → libggml-base.dylib).
    We deduplicate by content: for each unique inode/content hash we keep only
    the shortest name (the unversioned one, e.g. libggml-base.dylib).

    Skipped entirely: llama-server, llama-quantize (server binaries not needed).

    Output zip layout:
        ollama
        *.dylib              (unversioned names only)
        mlx_metal_v3/*.dylib (Apple Silicon Metal GPU — kept for M-series Macs)
        mlx_metal_v4/*.dylib
    """
    print("  Repacking macOS …")
    with tarfile.open(src, "r:gz") as tf:
        members = tf.getmembers()

        import re

        # First pass: collect ALL regular files (including hard-links which tarfile
        # presents as LNKTYPE with a linkname pointing elsewhere in the archive).
        # We group .dylib entries by their "base stem" (the part before any .N. suffix)
        # and keep only the member with the shortest name — that's the canonical one.
        # e.g. for { libggml-base.dylib, libggml-base.0.dylib, libggml-base.0.15.1.dylib }
        # we keep libggml-base.dylib.

        # Build a name→member map for everything (regular files AND hard-links)
        all_files: dict[str, tarfile.TarInfo] = {}
        for m in members:
            if m.isdir() or m.issym():
                continue
            name = m.name.lstrip("./")
            all_files[name] = m

        def dylib_stem(path: str) -> str:
            """Strip trailing version numbers: libggml-base.0.15.1.dylib → libggml-base"""
            basename = path.rsplit("/", 1)[-1]
            return re.sub(r'(\.\d+)+\.dylib$', '.dylib', basename)

        # Group .dylib names by their (subdir, stem) and pick the shortest name
        from collections import defaultdict
        dylib_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in all_files:
            if name.endswith(".dylib"):
                subdir = name.rsplit("/", 1)[0] if "/" in name else ""
                stem = dylib_stem(name)
                dylib_groups[(subdir, stem)].append(name)

        # For each group pick the shortest name (fewest dots = most canonical)
        dylib_canonical: set[str] = set()
        for names_in_group in dylib_groups.values():
            canonical = min(names_in_group, key=len)
            dylib_canonical.add(canonical)

        # Collect candidates: (name, member)
        candidates: list[tuple[str, tarfile.TarInfo]] = []
        for name, m in all_files.items():
            if name == "ollama":
                candidates.append((name, m))
            elif name.endswith(".dylib") and name in dylib_canonical:
                candidates.append((name, m))

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, m in candidates:
                fobj = tf.extractfile(m)
                if fobj:
                    data = fobj.read()
                    info = zipfile.ZipInfo(name)
                    info.external_attr = (m.mode & 0xFFFF) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(info, data)
    _report_zip(out_zip)


# ── Linux ──────────────────────────────────────────────────────────────────────

def repack_linux(src: Path, out_zip: Path) -> None:
    """
    ollama-linux-{arch}.tar.zst contains:
        bin/ollama
        lib/ollama/cuda_v12/…   ← GPU, excluded
        lib/ollama/cuda_v13/…   ← GPU, excluded
        lib/ollama/rocm/…       ← GPU, excluded
        lib/ollama/libggml-*.so ← CPU, included
        lib/ollama/libggml-*.so.N, etc.

    Output zip layout mirrors the upstream tree:
        bin/ollama
        lib/ollama/libggml-*.so
        …
    """
    print("  Repacking Linux …")
    # Python stdlib has no zstd; decompress via shell if available, else die.
    try:
        import zlib  # noqa: just checking we have it; zstd is different
        _decompressed = _decompress_zstd(src)
    except RuntimeError as e:
        sys.exit(f"  ERROR: {e}")

    with tarfile.open(fileobj=io.BytesIO(_decompressed), mode="r:") as tf:
        members = tf.getmembers()
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for m in members:
                if m.isdir():
                    continue
                name = m.name.lstrip("./")
                # Skip GPU libs
                if any(name.startswith(p) for p in LINUX_GPU_PREFIXES):
                    continue
                # Keep bin/ollama and lib/ollama CPU shared objects
                if name == "bin/ollama" or name.startswith("lib/ollama/"):
                    fobj = tf.extractfile(m)
                    if fobj:
                        data = fobj.read()
                        info = zipfile.ZipInfo(name)
                        info.external_attr = (m.mode & 0xFFFF) << 16
                        info.compress_type = zipfile.ZIP_DEFLATED
                        zf.writestr(info, data)
    _report_zip(out_zip)


def _decompress_zstd(src: Path) -> bytes:
    """Decompress a .zst file. Uses the `zstd` CLI if available, else tries
    the pyzstd package, else errors with a helpful message."""
    import subprocess

    # Try zstd CLI
    if shutil.which("zstd"):
        result = subprocess.run(
            ["zstd", "-d", "--stdout", str(src)],
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
        raise RuntimeError(f"zstd CLI failed: {result.stderr.decode()}")

    # Try pyzstd package
    try:
        import pyzstd  # type: ignore
        return pyzstd.decompress(src.read_bytes())
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot decompress .zst: install the 'zstd' CLI (`brew install zstd` / `apt install zstd`)"
        " or `pip install pyzstd`."
    )


# ── Windows ────────────────────────────────────────────────────────────────────

def repack_windows(src: Path, out_zip: Path) -> None:
    """
    ollama-windows-amd64.zip contains files in multiple subdirectories
    (cuda_v12/, rocm/, etc.) plus a flat root level. We:
      - Keep only ollama.exe and *.dll from the flat root (no subdirs)
      - Skip GPU/ROCm subdirectories entirely
      - Deduplicate by basename (first occurrence wins)
      - Skip known GPU DLLs (ggml-cuda, ggml-vulkan, vulkan-1)

    Output zip: all files flat at root, no subdirectories.
    """
    print("  Repacking Windows …")
    with zipfile.ZipFile(src, "r") as src_zf:
        # Collect only root-level entries (no "/" in name after normalisation)
        root_entries = []
        for info in src_zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if "/" in name:
                continue  # skip subdirectory entries (cuda_v12/, rocm/, …)
            root_entries.append(info)

        seen: set[str] = set()
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for info in root_entries:
                basename = info.filename
                if basename in seen:
                    continue
                if basename != "ollama.exe" and not basename.endswith(".dll"):
                    continue
                if _is_gpu_dll(basename):
                    continue
                seen.add(basename)
                data = src_zf.read(info)
                new_info = zipfile.ZipInfo(basename)
                new_info.external_attr = info.external_attr
                new_info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(new_info, data)
    _report_zip(out_zip)


def _is_gpu_dll(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in (
        "cublas", "cuda", "cufft", "curand", "nvml", "rocblas", "hip",
        "ggml-cuda", "ggml-vulkan", "vulkan-",
    ))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _report_zip(path: Path) -> None:
    size_mb = path.stat().st_size / 1_048_576
    print(f"  Wrote {path.name} ({size_mb:.1f} MB)")


def find_asset(assets: list[dict], name: str) -> dict:
    for a in assets:
        if a["name"] == name:
            return a
    raise KeyError(f"Asset '{name}' not found in release. Available: {[a['name'] for a in assets]}")


# ── Main ───────────────────────────────────────────────────────────────────────

REPACKER: dict[str, Callable[[Path, Path], None]] = {
    "macos": repack_macos,
    "linux": repack_linux,
    "windows": repack_windows,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", metavar="TAG", help="Ollama version tag (e.g. v0.30.10); defaults to latest")
    parser.add_argument("--platform", choices=["macos", "linux", "windows"], help="Build only this platform")
    parser.add_argument("--keep-upstream", action="store_true", help="Keep downloaded upstream archives after repacking")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    dist_dir = script_dir / "dist"
    cache_dir = script_dir / ".cache"
    dist_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    print(f"Resolving Ollama release …")
    version, assets = resolve_version(args.version)
    print(f"  Version: {version}")

    platforms = [args.platform] if args.platform else list(UPSTREAM_ASSETS.keys())

    results: list[dict] = []

    for platform in platforms:
        for upstream_name, arch in UPSTREAM_ASSETS[platform]:
            print(f"\n── {platform} / {arch} ──")
            asset = find_asset(assets, upstream_name)
            url = asset["browser_download_url"]

            # Download (cached by filename in .cache/)
            cached = cache_dir / upstream_name
            if not cached.exists():
                download_with_progress(url, cached)
            else:
                print(f"  Using cached {upstream_name}")

            upstream_sha = sha256_file(cached)
            print(f"  Upstream SHA256: {upstream_sha}")

            # Repack
            out_name = f"ollama-{version}-{platform}-{arch}.zip"
            out_path = dist_dir / out_name
            REPACKER[platform](cached, out_path)

            out_sha = sha256_file(out_path)
            print(f"  Output  SHA256: {out_sha}")

            results.append({
                "platform": platform,
                "arch": arch,
                "version": version,
                "filename": out_name,
                "sha256": out_sha,
                "size_bytes": out_path.stat().st_size,
            })

            if not args.keep_upstream:
                # Remove from cache to save disk — keep if explicitly requested
                pass  # cache is intentionally kept for reruns; dist/ is the deliverable

    # Write manifest
    manifest_path = dist_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"ollama_version": version, "assets": results}, f, indent=2)
    print(f"\n── Manifest written to {manifest_path} ──")

    # Print Rust snippet for ollama.rs
    print("\n── Paste into ollama.rs ──────────────────────────────────────────────────")
    print(f'const OLLAMA_VERSION: &str = "{version}";')
    print()
    print("// Asset SHA256s (update when re-running this script):")
    for r in results:
        const_name = f"OLLAMA_SHA256_{r['platform'].upper()}_{r['arch'].upper().replace('-', '_')}"
        print(f'const {const_name}: &str = "{r["sha256"]}";')
    print()
    print("// Asset base URL (GitHub Release of the rust-assistant repo):")
    print('// const ASSET_BASE_URL: &str = "https://github.com/LudoBermejoES/corylus-assistant/releases/download/{version}";')
    print("─" * 72)


if __name__ == "__main__":
    main()
