from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from test_prebuilt_inspection import bundle as bundle, fake_onnx as fake_onnx
from npu_model.cli.main import app
from npu_model.core.errors import NpuModelError
from npu_model.inspect.prebuilt import inventory
from npu_model.packagers import local_import as local


@pytest.fixture
def modelfile(bundle: Path) -> Path:
    path = bundle / "Modelfile"
    path.write_bytes(
        b'FROM .\r\n# Caller-supplied exact role tokens\r\nTEMPLATE """{{ .System }}\r\n{{ .Prompt }}"""\r\n'
        b'PARAMETER num_ctx 4096\r\nPARAMETER num_predict 7\r\nPARAMETER stop "exact-stop"\r\n'
        b'LICENSE """Original license\r\nunchanged"""\r\nSYSTEM "caller system"\r\n')
    (bundle / "LICENSE").write_bytes(b"Original license bytes")
    (bundle / "chat_template.jinja").write_text("{% for message in messages %}{{ message }}{% endfor %}")
    return path


def make_plan(bundle: Path, modelfile: Path, **kwargs: object) -> local.ImportPlan:
    options = {"executable": Path(sys.executable), "host": "http://127.0.0.1:18765", "modelfile": modelfile}
    options.update(kwargs)
    return local.plan_import(bundle, "local/fixture:c1", **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("host", [None, "http://localhost:11434", "http://example.invalid:11434",
                                 "https://127.0.0.1:1234", "http://127.0.0.1", "http://127.0.0.1:0",
                                 "http://secret@127.0.0.1:1234", "http://127.0.0.1:1234/path",
                                 "http://127.0.0.1:1234?token=secret", "http://127.0.0.1:1234#fragment"])
def test_endpoint_guards(host: str | None) -> None:
    with pytest.raises(NpuModelError):
        local.local_endpoint(host)


def test_numeric_ipv6_endpoint() -> None:
    assert local.local_endpoint("http://[::1]:18765/") == "http://[::1]:18765"


def test_plan_has_no_network_subprocess_or_source_writes(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = MagicMock(side_effect=AssertionError("no execution during plan"))
    monkeypatch.setattr(local, "_api", forbidden)
    monkeypatch.setattr(local, "_run_create", forbidden)
    before = inventory(bundle)
    plan = make_plan(bundle, modelfile)
    assert plan.endpoint.endswith(":18765")
    assert plan.to_dict()["executed"] is False
    assert plan.to_dict()["inspection"]["levels"]["native_evidence"]["certified"] is False
    assert inventory(bundle) == before
    forbidden.assert_not_called()


def test_cli_plan_and_environment_choices(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:18999")
    monkeypatch.setenv("NPU_MODEL_OLLAMA_EXE", sys.executable)
    result = CliRunner().invoke(app, ["import-ollama", "--input", str(bundle), "--name", "local/test:c1"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["host"].endswith(":18999") and value["executed"] is False
    assert "published" not in result.output.lower()


@pytest.mark.parametrize("source", ["remote-tag:latest", "../other", "https://example.invalid/model", "*.onnx"])
def test_local_from_guards(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, source: str,
) -> None:
    modelfile.write_text(f'FROM {source}\nTEMPLATE "{{{{ .Prompt }}}}"\n')
    with pytest.raises(NpuModelError) as error:
        make_plan(bundle, modelfile)
    assert error.value.reason_code == "LOCAL_FROM_REQUIRED"


@pytest.mark.parametrize("text,code", [
    ('FROM .\nFROM .\nTEMPLATE "{{ .Prompt }}"\n', "LOCAL_FROM_REQUIRED"),
    ('FROM .\n', "OLLAMA_TEMPLATE_REQUIRED"),
    ('FROM .\nTEMPLATE "{% for message in messages %}{{ message }}"\n', "OLLAMA_TEMPLATE_REQUIRED"),
    ('FROM .\nTEMPLATE "{{ .Prompt }}"\nDRAFT remote:latest\n', "UNSAFE_MODELFILE"),
    ('FROM .\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_ctx 8192\n', "CONTEXT_CONFLICT"),
    ('FROM .\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_ctx 0\n', "CONTEXT_CONFLICT"),
    ('FROM .\nTEMPLATE "{{ .Prompt }}"\nPARAMETER " "\n', "INVALID_MODELFILE"),
    ('FROM .\nTEMPLATE ".Prompt"\n', "OLLAMA_TEMPLATE_REQUIRED"),
    ('FROM .\nTEMPLATE """unterminated\n', "INVALID_MODELFILE"),
])
def test_modelfile_guards(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, text: str, code: str,
) -> None:
    modelfile.write_text(text)
    with pytest.raises(NpuModelError) as error:
        make_plan(bundle, modelfile)
    assert error.value.reason_code == code


def test_source_substitution_rejected(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path,
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    modelfile.write_text(f'FROM "{other}"\nTEMPLATE "{{{{ .Prompt }}}}"')
    with pytest.raises(NpuModelError) as error:
        make_plan(bundle, modelfile)
    assert error.value.reason_code == "SOURCE_SUBSTITUTION"


def test_invalid_input_never_invokes_create(
    bundle: Path, modelfile: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    create = MagicMock()
    monkeypatch.setattr(local, "_run_create", create)
    result = CliRunner().invoke(app, ["import-ollama", "--input", str(bundle), "--name", "test:c1",
        "--ollama-exe", sys.executable, "--host", "http://127.0.0.1:18765", "--execute"])
    assert result.exit_code == 2 and "INSPECTION_FAILED" in result.output
    create.assert_not_called()


def test_execution_requires_reviewed_fingerprint(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    tags = MagicMock(side_effect=AssertionError("no network without reviewed plan"))
    monkeypatch.setattr(local, "_tags", tags)
    with pytest.raises(NpuModelError) as error:
        local.execute_import(plan, expected_fingerprint=None, work_dir=tmp_path)
    assert error.value.reason_code == "PLAN_CHANGED"
    tags.assert_not_called()


@pytest.mark.parametrize("tamper", ["asset", "modelfile", "name", "endpoint", "overwrite"])
def test_plan_identity_binding(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tamper: str,
) -> None:
    first = make_plan(bundle, modelfile)
    options = {}
    if tamper == "asset":
        (bundle / "LICENSE").write_text("changed")
    elif tamper == "modelfile":
        modelfile.write_bytes(modelfile.read_bytes() + b'# changed\n')
    elif tamper == "endpoint":
        options["host"] = "http://127.0.0.1:18999"
    elif tamper == "overwrite":
        options["overwrite"] = True
    elif tamper == "name":
        second = local.plan_import(bundle, "other:c1", executable=Path(sys.executable),
                                   host=first.endpoint, modelfile=modelfile)
        assert first.fingerprint != second.fingerprint
        return
    assert first.fingerprint != make_plan(bundle, modelfile, **options).fingerprint


def test_existing_tag_guard_before_staging(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local, "_tags", lambda *args: {"local/fixture:c1"})
    create = MagicMock()
    monkeypatch.setattr(local, "_run_create", create)
    with pytest.raises(NpuModelError) as error:
        local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    assert error.value.reason_code == "TAG_EXISTS"
    create.assert_not_called()
    assert not list(tmp_path.glob("local-import-*"))


def test_post_plan_tamper_never_calls_create(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    (bundle / "context.qnn").write_bytes(b"tampered")
    monkeypatch.setattr(local, "_tags", lambda *args: set())
    create = MagicMock()
    monkeypatch.setattr(local, "_run_create", create)
    with pytest.raises(NpuModelError) as error:
        local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    assert error.value.reason_code == "ASSET_CHANGED"
    create.assert_not_called()
    assert not list(tmp_path.glob("local-import-*"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_fake_cli_import_preserves_modelfile_assets_and_endpoint(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, overwrite: bool,
) -> None:
    """Real subprocess helper; no Ollama server, ONNX validity or native execution claim."""
    plan = make_plan(bundle, modelfile, overwrite=overwrite)
    before = inventory(bundle)
    tags = MagicMock(side_effect=[{"local/fixture:c1"} if overwrite else set(),
                                 {"local/fixture:c1"} if overwrite else set(), {"local/fixture:c1"}])
    monkeypatch.setattr(local, "_tags", tags)
    capture = tmp_path / "capture.json"
    helper = Path(__file__).parent / "fixtures" / "fake_ollama.py"
    run = local._run_create
    def fake_create(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> None:
        run([sys.executable, str(helper), str(capture), *command[1:]], cwd, env, timeout)
    monkeypatch.setattr(local, "_run_create", fake_create)
    result = local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    value = json.loads(capture.read_text())
    assert result["installed"] and result["native_evidence"]["certified"] is False
    assert value["host"] == plan.endpoint and value["argv"][0:2] == ["create", plan.name]
    generated = value["modelfile"]
    assert generated[generated.index("\r\n"):] == modelfile.read_bytes().decode()[len("FROM ."):]
    assert "Original license\r\nunchanged" in generated and "exact-stop" in generated
    assert value["asset_hashes"] == {asset.path: asset.sha256 for asset in before}
    assert inventory(bundle) == before and not list(tmp_path.glob("local-import-*"))
    assert value["auth"] == "false" and value["no_proxy"] == "*"


@pytest.mark.parametrize("failure,code", [
    ("returncode", "CREATE_FAILED"), ("timeout", "CREATE_TIMEOUT"), ("launch", "CREATE_LAUNCH_FAILED"),
])
def test_subprocess_failures_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                      failure: str, code: str) -> None:
    import subprocess
    mock = MagicMock(return_value=SimpleNamespace(returncode=9))
    if failure == "timeout":
        mock.side_effect = subprocess.TimeoutExpired("create", 1)
    elif failure == "launch":
        mock.side_effect = OSError("cannot execute")
    monkeypatch.setattr(local.subprocess, "run", mock)
    with pytest.raises(NpuModelError) as error:
        local._run_create(["fake", "create"], tmp_path, {}, 1)
    assert error.value.reason_code == code
    assert mock.call_args.kwargs["shell"] is False


def test_create_zero_without_tag_is_not_success(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local, "_tags", lambda *args: set())
    create = MagicMock()
    monkeypatch.setattr(local, "_run_create", create)
    with pytest.raises(NpuModelError) as error:
        local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    assert error.value.reason_code == "CREATE_NOT_CONFIRMED"
    create.assert_called_once()
    assert not list(tmp_path.glob("local-import-*"))


def test_tag_queries_honor_execution_budget(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local.time, "monotonic", lambda: 100.0)
    queries = 0

    def tags(endpoint: str, timeout: float) -> set[str]:
        nonlocal queries
        assert timeout == 300.0  # Large verified ORT inventories can take more than 15 seconds.
        queries += 1
        return {plan.name} if queries == 3 else set()

    monkeypatch.setattr(local, "_tags", tags)
    monkeypatch.setattr(local, "_run_create", MagicMock())
    result = local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    assert result["installed"] is True and queries == 3


def test_confirmation_error_warns_about_existing_tag(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local, "_tags", MagicMock(side_effect=[
        set(), set(), NpuModelError("import", "LOCAL_SERVER_UNAVAILABLE", "timed out")]))
    monkeypatch.setattr(local, "_run_create", MagicMock())
    with pytest.raises(NpuModelError) as caught:
        local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
    assert caught.value.reason_code == "CREATE_NOT_CONFIRMED"
    assert "may already exist" in str(caught.value)
    assert not list(tmp_path.glob("local-import-*"))


def test_app_executable_guard(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path,
) -> None:
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    exe = standalone / "ollama.exe"
    exe.write_bytes(b"fake-executable")
    (standalone / "ollama app.exe").write_bytes(b"fake-app")
    with pytest.raises(NpuModelError) as error:
        make_plan(bundle, modelfile, executable=exe)
    assert error.value.reason_code == "AUTO_START_RISK"


def test_input_work_directory_rejected(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local, "_tags", lambda *args: set())
    with pytest.raises(NpuModelError) as error:
        local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=bundle)
    assert error.value.reason_code == "INVALID_WORK_PATH"
    assert not list(bundle.glob("local-import-*"))

def test_cleanup_failure_does_not_mask_create_failure(
    bundle: Path, modelfile: Path, fake_onnx: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan(bundle, modelfile)
    monkeypatch.setattr(local, "_tags", lambda *args: set())
    monkeypatch.setattr(local, "_run_create", MagicMock(side_effect=NpuModelError("import", "CREATE_FAILED", "failed")))
    cleanup = local.shutil.rmtree
    monkeypatch.setattr(local.shutil, "rmtree", MagicMock(side_effect=PermissionError("fixture cleanup denied")))
    try:
        with pytest.raises(NpuModelError) as error:
            local.execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=tmp_path)
        assert error.value.reason_code == "CREATE_FAILED"
        assert "Remove it manually" in error.value.hint
    finally:
        for workspace in tmp_path.glob("local-import-*"):
            cleanup(workspace)


def test_child_environment_has_no_proxy_or_auth_leaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "OLLAMA_API_KEY", "HF_TOKEN", "OLLAMA_AUTH"):
        monkeypatch.setenv(key, "secret")
    env = local._child_env(Path(sys.executable), tmp_path, "http://127.0.0.1:18999")
    assert not any(key in env for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "OLLAMA_API_KEY", "HF_TOKEN"))
    assert env["OLLAMA_AUTH"] == "false" and env["USERPROFILE"] == str(tmp_path)
    assert env["PATH"] != os.environ.get("PATH")


def test_api_no_redirect_auth_proxy_and_bounded_response(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = MagicMock()
    connection.getresponse.return_value.status = 302
    connection.getresponse.return_value.read.return_value = b'{"models":[]}'
    constructor = MagicMock(return_value=connection)
    monkeypatch.setattr(local.http.client, "HTTPConnection", constructor)
    with pytest.raises(NpuModelError) as error:
        local._api("http://127.0.0.1:18999", "/api/tags", 1)
    assert error.value.reason_code == "LOCAL_SERVER_ERROR"
    constructor.assert_called_once_with("127.0.0.1", 18999, timeout=1)
    assert connection.request.call_args.kwargs["headers"] == {"Accept": "application/json"}
    connection.getresponse.return_value.read.assert_called_once_with(local.MAX_RESPONSE + 1)
    connection.close.assert_called_once()
