from __future__ import annotations

from pathlib import Path

import pytest

from npu_model.core.cache import ConversionCache, compute_cache_key


def test_compute_cache_key_deterministic() -> None:
    params = dict(
        input_spec="hf:microsoft/Phi-3-mini",
        input_revision="main",
        adapter_id="phi3",
        mode="prebuilt-ort-genai",
        backend_id="qnn",
        target_name="auto",
        target_params={"backend_type": "htp"},
        compile_strategy="passthrough",
        compile_config={},
        quantizer_id="passthrough",
        runtime_format_id="ort-genai-folder",
        tool_version="0.1.0",
    )
    key1 = compute_cache_key(**params)
    key2 = compute_cache_key(**params)
    assert key1 == key2
    assert len(key1) == 24


def test_compute_cache_key_changes_with_input() -> None:
    base = dict(
        input_spec="hf:microsoft/Phi-3-mini",
        input_revision="main",
        adapter_id="phi3",
        mode="prebuilt-ort-genai",
        backend_id="qnn",
        target_name="auto",
        target_params={},
        compile_strategy="passthrough",
        compile_config={},
        quantizer_id="passthrough",
        runtime_format_id="ort-genai-folder",
        tool_version="0.1.0",
    )
    key1 = compute_cache_key(**base)
    key2 = compute_cache_key(**{**base, "input_spec": "hf:other/model"})
    assert key1 != key2


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ConversionCache(tmp_path)
    assert cache.get("nonexistent") is None


@pytest.mark.parametrize("option", ["precision", "execution_provider", "calib_samples"])
def test_cache_key_includes_export_and_calibration_settings(option: str) -> None:
    base = dict(
        input_spec="local-model", input_revision=None, adapter_id="phi3", mode="export",
        backend_id="qnn", target_name="auto", target_params={}, compile_strategy="passthrough",
        compile_config={}, quantizer_id="passthrough", runtime_format_id="ort-genai-folder",
        tool_version="0.1.0",
    )
    assert compute_cache_key(**base, export_options={option: "a"}) != compute_cache_key(
        **base, export_options={option: "b"},
    )
    assert compute_cache_key(**base, shape_options={"num_ctx": 512}) != compute_cache_key(
        **base, shape_options={"num_ctx": 1024},
    )


def test_cache_put_and_get(tmp_path: Path) -> None:
    cache = ConversionCache(tmp_path)

    # Create fake bundle + manifest
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"\x00")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"test": true}', encoding="utf-8")

    entry = cache.put("testkey", bundle, manifest, meta={"note": "test"})
    assert entry.valid

    # Now get it back
    retrieved = cache.get("testkey")
    assert retrieved is not None
    assert retrieved.valid
    assert retrieved.key == "testkey"


def test_cache_restore(tmp_path: Path) -> None:
    cache = ConversionCache(tmp_path / "work")

    # Create fake bundle + manifest
    bundle = tmp_path / "bundle_src"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"\x00")
    (bundle / "config.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest_src.json"
    manifest.write_text('{"v": 1}', encoding="utf-8")

    cache.put("k1", bundle, manifest, meta={})
    entry = cache.get("k1")
    assert entry is not None

    out_dir = tmp_path / "restored"
    out_dir.mkdir()
    bundle_dst, manifest_dst = cache.restore(entry, out_dir)
    assert bundle_dst.exists()
    assert manifest_dst.exists()
    assert (bundle_dst / "model.onnx").exists()
