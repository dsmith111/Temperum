from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from npu_model.cli.main import app
from npu_model.core.errors import NpuModelError
from npu_model.inspect.prebuilt import content_fingerprint, inspect_prebuilt, inventory
from npu_model.packagers import prebuilt_profile as profile
from npu_model.packagers.local_import import read_modelfile
from test_prebuilt_inspection import Proto, context_node, fake_onnx as fake_onnx, graph_model  # noqa: F401


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_onnx: SimpleNamespace) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    stages: dict[str, Any] = {}
    for index, (filename, provider) in enumerate(profile.STAGES):
        (root / filename).write_bytes(b"stub-graph-policy-test-only")
        stage: dict[str, Any] = {"filename": filename}
        if provider == "qnn":
            stage["session_options"] = {"provider_options": [{"qnn": {"soc_model": "60"}}]}
        stages[str(3 - index)] = stage
    (root / "context.qnn").write_bytes(b"stub-context-never-native-certified")
    (root / "README.md").write_bytes(b"fixture upstream card, unchanged\n")
    (root / "modeling_phi3.py").write_bytes(b"raise RuntimeError('MUST NOT RUN')\n")
    (root / "tokenizer.json").write_text('{"model":{}}')
    config = {"model": {"type": "decoder-pipeline", "context_length": 4096, "vocab_size": 10,
                         "eos_token_id": 2, "decoder": {"session_options": {"provider_options": [], "log_id": "original"},
                                                       "pipeline": [stages]}},
              "search": {"max_length": 4096}}
    (root / "genai_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(profile, "SOURCE_FINGERPRINT", content_fingerprint(inventory(root)))

    def load(path: str, **kwargs: Any) -> Proto:
        if Path(path).name in {"context_ctx.onnx", "iterator_ctx.onnx"}:
            return graph_model(context_node())
        return graph_model(Proto(op_type="Identity", domain="", attribute=[]))

    fake_onnx.load.side_effect = load
    assert inspect_prebuilt(root).importable
    return root


def plan(source: Path, out: Path | None = None) -> profile.ProfilePlan:
    return profile.plan_profile(source, out or source.parent / "derived", profile=profile.PROFILE)


def execute(value: profile.ProfilePlan, **kwargs: Any) -> dict[str, Any]:
    return profile.execute_profile(value, expected_fingerprint=value.fingerprint, accept_license=True, **kwargs)


def error(code: str, fn: Any) -> None:
    with pytest.raises(NpuModelError) as caught:
        fn()
    assert caught.value.reason_code == code, caught.value


def test_plan_readonly_and_deterministic(source: Path) -> None:
    before = inventory(source)
    value = plan(source)
    assert value.to_dict()["executed"] is False
    assert value.fingerprint == plan(source).fingerprint
    assert inventory(source) == before and not Path(value.destination).exists()
    assert value.fingerprint != plan(source, source.parent / "different").fingerprint


def test_derivative_preserves_and_notices(source: Path) -> None:
    before = inventory(source)
    value = plan(source)
    result = execute(value)
    out = Path(value.destination)
    assert result["executed"] is True and result["native_evidence"] == "not-run"
    assert inventory(source) == before
    for asset in before:
        if asset.path != "genai_config.json":
            assert (out / asset.path).read_bytes() == (source / asset.path).read_bytes()
    original = json.loads((source / "genai_config.json").read_bytes())
    derived = json.loads((out / "genai_config.json").read_bytes())
    assert [list(group) for group in derived["model"]["decoder"]["pipeline"]] == [
        list(group) for group in original["model"]["decoder"]["pipeline"]]
    for group in derived["model"]["decoder"]["pipeline"]:
        for stage in group.values():
            for provider in stage.get("session_options", {}).get("provider_options", []):
                assert provider["qnn"].pop("disable_file_mapped_weights") == "1"
    assert derived["model"]["decoder"]["session_options"]["log_id"] == profile.MODIFIED_LOG_ID
    derived["model"]["decoder"]["session_options"]["log_id"] = original["model"]["decoder"]["session_options"]["log_id"]
    assert derived == original
    assert (out / "source" / "genai_config.json").read_bytes() == (source / "genai_config.json").read_bytes()
    inspected = inspect_prebuilt(out, processor_name=profile.PROCESSOR)
    assert inspected.importable and inspected.target["status"] == "declared-match"
    assert inspected.declaration["source"]["revision"] == profile.REVISION
    assert inspected.declaration["context_length"] == 4096
    text = (out / "Modelfile").read_bytes()
    assert b"\r" not in text and b"PARAMETER num_ctx 512\n" in text
    assert profile.TEMPLATE.encode() in text
    parsed = read_modelfile(out / "Modelfile", out, inspected.context_length)
    assert "LOCAL CHANGES" in parsed.text and "Apache License" in parsed.text and "MIT License" in parsed.text
    assert "base weights" in (out / "NOTICE.txt").read_text()
    assert json.loads((out / "source_receipt.json").read_bytes())["source"]["assets"]
    assert not list(out.parent.glob("npu-profile-*"))


def test_wrong_source_and_unknown_profile(source: Path) -> None:
    error("UNKNOWN_PROFILE", lambda: profile.plan_profile(source, source.parent / "new", profile="guessed"))
    (source / "README.md").write_bytes(b"changed")
    error("SOURCE_MISMATCH", lambda: plan(source))


def test_fingerprint_and_consent(source: Path) -> None:
    value = plan(source)
    error("PLAN_CHANGED", lambda: profile.execute_profile(value, expected_fingerprint="0" * 64, accept_license=True))
    error("RIGHTS_CONSENT_REQUIRED",
          lambda: profile.execute_profile(value, expected_fingerprint=value.fingerprint, accept_license=False))
    assert not Path(value.destination).exists()
    (source / "modeling_phi3.py").write_bytes(b"tampered")
    error("SOURCE_MISMATCH", lambda: execute(value))


def test_existing_output_and_nested_output(source: Path) -> None:
    error("INVALID_DESTINATION", lambda: plan(source, source / "nested"))
    out = source.parent / "derived"
    out.mkdir()
    marker = out / "user.txt"
    marker.write_bytes(b"preserve")
    error("DESTINATION_EXISTS", lambda: plan(source))
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("failure", ["copy", "promotion", "validation", "collision"])
def test_failures_cleanup_owned_only(source: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    value = plan(source)
    expected = "LOCAL_IO_FAILED"
    if failure == "copy":
        def fail_copy(*args: Any) -> None:
            raise OSError("synthetic copy failure")
        monkeypatch.setattr(profile.shutil, "copyfile", fail_copy)
    elif failure == "promotion":
        original = Path.rename

        def fail_rename(path: Path, dest: Path) -> Path:
            if path.name.startswith("npu-profile-"):
                raise OSError("synthetic promotion failure")
            return original(path, dest)
        monkeypatch.setattr(Path, "rename", fail_rename)
    elif failure == "validation":
        generated = dict(value.generated)
        generated["ollama_npu.json"] = b"{}"
        value = replace(value, generated=tuple(sorted(generated.items())))
        expected = "PLAN_CHANGED"
    else:
        monkeypatch.setattr(profile.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
        collision = source.parent / "npu-profile-fixed"
        collision.mkdir()
        (collision / "user.txt").write_bytes(b"preserve")
    error(expected, lambda: execute(value))
    assert not Path(value.destination).exists()
    remaining = list(source.parent.glob("npu-profile-*"))
    if failure == "collision":
        assert (remaining[0] / "user.txt").read_bytes() == b"preserve"
    else:
        assert not remaining


def test_cli_profile_boundary(source: Path) -> None:
    args = ["prepare-prebuilt", "--input", str(source), "--out", str(source.parent / "cli-derived"),
            "--profile", profile.PROFILE]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    result = CliRunner().invoke(app, args + ["--execute", "--expect-fingerprint", value["fingerprint"], "--accept-license"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["executed"] is True
