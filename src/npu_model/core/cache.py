"""Deterministic caching for expensive pipeline stages."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _stable_json(obj: Any) -> str:
    """JSON-serialize with sorted keys for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, default=str)


def compute_cache_key(
    *,
    input_spec: str,
    input_revision: str | None,
    adapter_id: str,
    mode: str,
    backend_id: str,
    target_name: str,
    target_params: dict[str, Any],
    compile_strategy: str,
    compile_config: dict[str, Any],
    quantizer_id: str,
    runtime_format_id: str,
    tool_version: str,
    export_options: dict[str, Any] | None = None,
    shape_options: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 cache key from conversion parameters."""
    key_data = _stable_json({
        "input_spec": input_spec,
        "input_revision": input_revision,
        "adapter_id": adapter_id,
        "mode": mode,
        "backend_id": backend_id,
        "target_name": target_name,
        "target_params": target_params,
        "compile_strategy": compile_strategy,
        "compile_config": compile_config,
        "quantizer_id": quantizer_id,
        "runtime_format_id": runtime_format_id,
        "tool_version": tool_version,
        "export_options": export_options or {},
        "shape_options": shape_options or {},
    })
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheEntry:
    key: str
    cache_dir: Path
    bundle_dir: Path
    manifest_path: Path

    @property
    def valid(self) -> bool:
        return (
            self.cache_dir.is_dir()
            and self.bundle_dir.is_dir()
            and self.manifest_path.is_file()
        )


class ConversionCache:
    """File-system–based cache for conversion results.

    Layout::

        <work_dir>/.cache/<key>/
            bundle/ ...
            manifest.json
            cache_meta.json   # key params for debugging
    """

    def __init__(self, work_dir: Path) -> None:
        self.root = work_dir / ".cache"

    def get(self, key: str) -> CacheEntry | None:
        entry_dir = self.root / key
        entry = CacheEntry(
            key=key,
            cache_dir=entry_dir,
            bundle_dir=entry_dir / "bundle",
            manifest_path=entry_dir / "manifest.json",
        )
        return entry if entry.valid else None

    def put(
        self,
        key: str,
        bundle_dir: Path,
        manifest_path: Path,
        meta: dict[str, Any],
    ) -> CacheEntry:
        entry_dir = self.root / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Copy bundle
        dst_bundle = entry_dir / "bundle"
        if dst_bundle.exists():
            shutil.rmtree(dst_bundle)
        shutil.copytree(bundle_dir, dst_bundle)

        # Copy manifest
        dst_manifest = entry_dir / "manifest.json"
        shutil.copy2(manifest_path, dst_manifest)

        # Write cache metadata for debugging
        (entry_dir / "cache_meta.json").write_text(
            json.dumps({"key": key, **meta}, indent=2, default=str),
            encoding="utf-8",
        )

        return CacheEntry(
            key=key,
            cache_dir=entry_dir,
            bundle_dir=dst_bundle,
            manifest_path=dst_manifest,
        )

    def restore(self, entry: CacheEntry, out_dir: Path) -> tuple[Path, Path]:
        """Copy cached bundle + manifest into out_dir. Returns (bundle_dir, manifest_path)."""
        bundle_dst = out_dir / "ort_genai"
        if bundle_dst.exists():
            shutil.rmtree(bundle_dst)
        shutil.copytree(entry.bundle_dir, bundle_dst)

        manifest_dst = out_dir / "manifest.json"
        shutil.copy2(entry.manifest_path, manifest_dst)

        return bundle_dst, manifest_dst
