from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
from typer.testing import CliRunner

from npu_model.cli.main import app
from npu_model.core.errors import NpuModelError
from npu_model.inspect.prebuilt import inspect_prebuilt, inventory
from npu_model.sources import prebuilt_hf as hf
from test_prebuilt_inspection import bundle as bundle, fake_onnx as fake_onnx  # noqa: F401

COMMIT = "a" * 40
REPO = "public/prebuilt"


def blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data: bytes, *, status: int = 200, headers: dict[str, str] | None = None,
                 interruption: BaseException | None = None) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.interruption = interruption
        self.calls = 0

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def read(self, size: int | None = -1) -> bytes:
        self.calls += 1
        if self.interruption:
            if self.calls == 2:
                raise self.interruption
            size = min(size if size is not None else -1, 2)
        return super().read(size)

    def read1(self, size: int | None = -1) -> bytes:
        return self.read(size)


class Transport:
    """Synthetic HTTPS seam: exercises real transfer/hash/state code, never a server."""
    def __init__(self, monkeypatch: pytest.MonkeyPatch, source: Path) -> None:
        self.data = {asset.path: (source / asset.path).read_bytes() for asset in inventory(source)}
        self.data["LICENSE"] = b"fixture license text; no license approval"
        self.metadata: dict[str, Any] = {}
        self.refresh()
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.downloads: list[str] = []
        self.interrupt_path: str | None = None
        self.interruption: BaseException = KeyboardInterrupt()
        self.corrupt: str | None = None
        self.extra: bytes = b""
        self.length_override: str | None = None
        self.metadata_status = 200
        self.metadata_body: bytes | None = None
        self.redirect: str | None = None
        self.drift_after_download = False
        transport = self

        class Connection:
            def __init__(self, host: str, timeout: float) -> None:
                assert 0 < timeout <= 30
                self.host = host
                self.path = ""

            def request(self, method: str, path: str, headers: dict[str, str]) -> None:
                assert method == "GET"
                assert set(headers) == {"Accept-Encoding", "User-Agent"}
                transport.requests.append((self.host, path, headers))
                self.path = path

            def getresponse(self) -> Response:
                if self.path.startswith("/api/models/"):
                    meta = copy.deepcopy(transport.metadata)
                    if transport.drift_after_download and transport.downloads:
                        meta["siblings"][0]["size"] += 1
                    data = transport.metadata_body
                    return Response(data if data is not None else json.dumps(meta).encode(),
                                    status=transport.metadata_status)
                assert f"/resolve/{COMMIT}/" in self.path
                name = unquote(self.path.split(f"/resolve/{COMMIT}/", 1)[1].split("?", 1)[0])
                if transport.redirect:
                    return Response(b"", status=302, headers={"Location": transport.redirect})
                transport.downloads.append(name)
                data = transport.data[name]
                if transport.corrupt == name:
                    data = bytes([data[0] ^ 1]) + data[1:]
                data += transport.extra
                interruption = transport.interruption if transport.interrupt_path == name else None
                return Response(data, headers={"Content-Length": transport.length_override or str(len(data))},
                                interruption=interruption)

            def close(self) -> None:
                pass

        monkeypatch.setattr(hf.http.client, "HTTPSConnection", Connection)

    def refresh(self) -> None:
        siblings = []
        for name, data in sorted(self.data.items()):
            item: dict[str, Any] = {"rfilename": name, "size": len(data), "blobId": blob(data)}
            if name.endswith((".qnn", ".bin")):
                item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                item["blobId"] = "b" * 40  # Pointer-blob ID is NOT the content SHA256.
            siblings.append(item)
        self.metadata = {"id": REPO, "sha": COMMIT, "private": False, "gated": False, "siblings": siblings}


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch, bundle: Path, fake_onnx: Any) -> Transport:
    return Transport(monkeypatch, bundle)


def plan(tmp_path: Path, **kwargs: Any) -> hf.AcquisitionPlan:
    return hf.plan_acquisition(REPO, tmp_path / "acquired", work_dir=tmp_path, **kwargs)


def execute(value: hf.AcquisitionPlan, **kwargs: Any) -> dict[str, Any]:
    return hf.execute_acquisition(value, expected_fingerprint=value.fingerprint, accept_license=True, **kwargs)


def receipt(tmp_path: Path) -> Path:
    return next(tmp_path.glob("hf-acquire-*/receipt.json"))


def assert_error(code: str, fn: Any) -> NpuModelError:
    with pytest.raises(NpuModelError) as caught:
        fn()
    assert caught.value.reason_code == code, caught.value
    return caught.value


def test_plan_metadata_only(tmp_path: Path, remote: Transport) -> None:
    value = plan(tmp_path)
    assert value.commit == COMMIT and value.revision == "main"
    assert not remote.downloads and not list(tmp_path.glob("hf-acquire-*"))
    assert len(value.files) == len(remote.data)
    result = value.to_dict()
    assert result["executed"] is False and result["plan"]["policy"]["overwrite"] is False
    assert value.fingerprint == plan(tmp_path).fingerprint
    assert value.fingerprint != plan(tmp_path, processor_name="Exact Engineering CPU").fingerprint
    assert value.fingerprint != hf.plan_acquisition(REPO, tmp_path / "other", work_dir=tmp_path).fingerprint


def test_fingerprint_order_and_git_pointer_identity(tmp_path: Path, remote: Transport) -> None:
    value = plan(tmp_path)
    remote.metadata["siblings"].reverse()
    remote.metadata["downloads"] = 123
    assert plan(tmp_path).fingerprint == value.fingerprint
    for file in value.files:
        item = next(item for item in remote.metadata["siblings"] if item["rfilename"] == file.remote_path)
        assert file.blob_id == item["blobId"]
        assert file.oid == (item["lfs"]["sha256"] if "lfs" in item else item["blobId"])
    next(item for item in remote.metadata["siblings"] if "lfs" in item)["blobId"] = "c" * 40
    assert plan(tmp_path).fingerprint != value.fingerprint
    assert_error("METADATA_DRIFT", lambda: execute(value))
    assert not remote.downloads


def test_serialized_plan_does_not_mutate_fingerprint(tmp_path: Path, remote: Transport) -> None:
    value = plan(tmp_path)
    fingerprint = value.fingerprint
    serialized = value.to_dict()
    serialized["plan"]["policy"]["consumer"]["ort_version"] = "not-the-consumer"
    serialized["plan"]["files"][0]["oid"] = "0" * 40
    assert value.fingerprint == fingerprint
    assert plan(tmp_path).fingerprint == fingerprint


def test_explicit_commit_cannot_be_substituted(tmp_path: Path, remote: Transport) -> None:
    remote.metadata["sha"] = "b" * 40
    assert_error("METADATA_DRIFT", lambda: plan(tmp_path, revision=COMMIT))
    assert len(remote.requests) == 1 and not remote.downloads


@pytest.mark.parametrize("field,value", [
    ("sha", None), ("sha", "A" * 40), ("id", "other/repository"),
])
def test_invalid_repository_metadata(tmp_path: Path, remote: Transport, field: str, value: Any) -> None:
    remote.metadata[field] = value
    assert_error("INVALID_METADATA", lambda: plan(tmp_path))
    assert not remote.downloads


@pytest.mark.parametrize("body", [
    b'{"id":"public/prebuilt","id":"public/prebuilt"}',
    b'{"id":"public/prebuilt","value":NaN}',
    b'{"id":"public/prebuilt","value":Infinity}',
])
def test_invalid_metadata_json(tmp_path: Path, remote: Transport, body: bytes) -> None:
    remote.metadata_body = body
    assert_error("INVALID_METADATA", lambda: plan(tmp_path))
    assert not remote.downloads


@pytest.mark.parametrize("change", ["missing-blob", "invalid-blob", "null-lfs", "bool-size", "wrong-lfs-size"])
def test_lfs_metadata_requires_both_identities(tmp_path: Path, remote: Transport, change: str) -> None:
    item = next(item for item in remote.metadata["siblings"] if "lfs" in item)
    if change == "missing-blob":
        del item["blobId"]
    elif change == "invalid-blob":
        item["blobId"] = "b" * 64
    elif change == "null-lfs":
        item["lfs"] = None
    elif change == "bool-size":
        item["lfs"]["size"] = True
    else:
        item["lfs"]["size"] += 1
    assert_error("INVALID_METADATA", lambda: plan(tmp_path))
    assert not remote.downloads


def test_execute_independent(tmp_path: Path, remote: Transport) -> None:
    value = plan(tmp_path)
    result = execute(value)
    destination = Path(value.destination)
    assert result["stage"] == "completed" and len(remote.downloads) == len(remote.data)
    assert inspect_prebuilt(destination).importable
    assert (destination / "LICENSE").read_bytes() == remote.data["LICENSE"]
    state = hf._read_state(receipt(tmp_path))
    assert state["stage"] == "completed" and state["error"] is None and state["rights_consent"] is True
    assert len(state["completed"]) == len(value.files)
    assert result["inspection"]["levels"]["native_evidence"]["certified"] is False
    for file in value.files:
        path = destination / file.path
        assert not path.is_symlink()
        assert path.read_bytes() == remote.data[file.remote_path]
    remote.data.clear()
    assert (destination / "context.qnn").read_bytes()
    assert not (receipt(tmp_path).parent / "bundle").exists()
    assert not (receipt(tmp_path).parent / "lock").exists()


def test_python_sources_are_inert_and_preserved(tmp_path: Path, remote: Transport) -> None:
    marker = tmp_path / "must-not-exist"
    malicious = (f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
                 "raise RuntimeError('remote code must never execute')\n").encode()
    sources = {
        "configuration_phi3.py": malicious,
        "modeling_phi3.py": b"\xef\xbb\xbf# retained UTF-8 BOM and attribution\r\n" + malicious,
        "sitecustomize.py": malicious,
        "package/__init__.py": malicious,
        "not_valid_syntax.py": b"not valid Python source! preserve, do not compile\n",
    }
    remote.data.update(sources)
    remote.data["config.json"] = json.dumps({
        "auto_map": {"AutoConfig": "configuration_phi3.Config", "AutoModel": "modeling_phi3.Model"},
        "trust_remote_code": True,
    }).encode()
    remote.refresh()
    value = plan(tmp_path)
    policy = value.to_dict()["plan"]["policy"]["python_source"]
    assert policy["max_bytes"] == 4 << 20
    assert "never imported, compiled or executed" in policy["handling"]
    assert set(sources) <= {file.path for file in value.files}
    assert not marker.exists() and not remote.downloads
    execute(value)
    for name, data in sources.items():
        assert (Path(value.destination) / name).read_bytes() == data
    assert not marker.exists()
    assert not list(Path(value.destination).rglob("__pycache__"))
    completed = hf._read_state(receipt(tmp_path))["completed"]
    assert set(sources) <= {file["path"] for file in completed}
    execute(plan(tmp_path, resume=receipt(tmp_path)), resume=receipt(tmp_path))
    assert not marker.exists()


@pytest.mark.parametrize("size,valid", [(0, True), (4 << 20, True), ((4 << 20) + 1, False)])
def test_python_source_size_bound(tmp_path: Path, remote: Transport, size: int, valid: bool) -> None:
    remote.data["model.py"] = b"# source"
    remote.refresh()
    next(item for item in remote.metadata["siblings"] if item["rfilename"] == "model.py")["size"] = size
    if valid:
        assert next(file for file in plan(tmp_path).files if file.path == "model.py").size == size
    else:
        assert_error("INVENTORY_LIMIT", lambda: plan(tmp_path))
    assert not remote.downloads


@pytest.mark.parametrize("data", [b"\xff\xfe\x00source", b"print('text')\0"])
def test_binary_python_source_fails_without_exclusion(tmp_path: Path, remote: Transport, data: bytes) -> None:
    remote.data["model.py"] = data
    remote.refresh()
    value = plan(tmp_path)
    assert_error("UNSUPPORTED_SOURCE", lambda: execute(value))
    assert "model.py" in remote.downloads
    assert not Path(value.destination).exists()
    state = hf._read_state(receipt(tmp_path))
    assert state["partial"]["path"] == "model.py"
    assert state["error"]["code"] == "UNSUPPORTED_SOURCE"


def test_subfolder_preserves_ancestors(tmp_path: Path, remote: Transport) -> None:
    license_data = remote.data.pop("LICENSE")
    remote.data = {"exports/prebuilt/" + name: data for name, data in remote.data.items()}
    remote.data.update({"LICENSE": license_data, "README.md": b"card", "exports/NOTICE.txt": b"attribution",
                        "unselected/model.safetensors": b"unselected"})
    remote.refresh()
    value = plan(tmp_path, subfolder="exports/prebuilt")
    execute(value)
    destination = Path(value.destination)
    assert (destination / "_hf_repository" / "LICENSE").read_bytes() == license_data
    assert (destination / "_hf_repository" / "exports" / "NOTICE.txt").read_bytes() == b"attribution"
    assert (destination / "_hf_repository" / "README.md").read_bytes() == b"card"
    assert not any("unselected" in name for name in remote.downloads)
    assert inspect_prebuilt(destination).importable


def test_source_attribution_is_not_silently_excluded(tmp_path: Path, remote: Transport) -> None:
    remote.data = {"export/" + name: data for name, data in remote.data.items()}
    remote.data["NOTICE.py"] = b"# source attribution, inert and preserved\n"
    remote.refresh()
    value = plan(tmp_path, subfolder="export")
    execute(value)
    assert (Path(value.destination) / "_hf_repository" / "NOTICE.py").read_bytes() == remote.data["NOTICE.py"]


@pytest.mark.parametrize("failure,code", [(KeyboardInterrupt(), "CANCELLED"),
                                        (TimeoutError(), "TRANSFER_FAILED")])
def test_interruption_resume(tmp_path: Path, remote: Transport, failure: BaseException, code: str) -> None:
    value = plan(tmp_path)
    remote.interrupt_path = "model_ctx.onnx"
    remote.interruption = failure
    assert_error(code, lambda: execute(value))
    state_path = receipt(tmp_path)
    state = hf._read_state(state_path)
    assert state["stage"] in {"interrupted", "failed"} and state["error"]["code"] == code
    assert state["partial"]["size"] == 2 and state["completed"]
    completed = {item["path"] for item in state["completed"]}
    assert not Path(value.destination).exists()
    remote.interrupt_path = None
    remote.downloads.clear()
    # Simulate moving main: resume requests only the pinned commit.
    request_start = len(remote.requests)
    resumed = plan(tmp_path, resume=state_path)
    assert resumed.fingerprint == value.fingerprint
    result = execute(resumed, resume=state_path)
    assert result["stage"] == "completed"
    assert not completed & set(remote.downloads)
    assert "model_ctx.onnx" in remote.downloads
    assert all(f"/revision/{COMMIT}" in path for _, path, _ in remote.requests[request_start:]
               if path.startswith("/api/models/"))
    before = len(remote.downloads)
    execute(plan(tmp_path, resume=state_path), resume=state_path)
    assert len(remote.downloads) == before  # Completed receipts are reverified, not blindly trusted.


@pytest.mark.parametrize("after", [False, True])
def test_file_rename_interruption_resume(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch,
                                         after: bool) -> None:
    value = plan(tmp_path)
    original = Path.rename
    interrupted = False

    def interrupt(source: Path, target: Path) -> Path:
        nonlocal interrupted
        if source.name == "partial" and not interrupted:
            interrupted = True
            if after:
                original(source, target)
            raise KeyboardInterrupt()
        return original(source, target)

    monkeypatch.setattr(Path, "rename", interrupt)
    assert_error("CANCELLED", lambda: execute(value))
    state_path = receipt(tmp_path)
    state = hf._read_state(state_path)
    assert len(state["completed"]) == (1 if after else 0)
    assert (state["partial"] is None) is after
    completed = {item["path"] for item in state["completed"]}
    remote.downloads.clear()
    execute(plan(tmp_path, resume=state_path), resume=state_path)
    assert not completed & set(remote.downloads)
    assert hf._read_state(state_path)["stage"] == "completed"


@pytest.mark.parametrize("after", [False, True])
def test_receipt_write_interruption_resume(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch,
                                            after: bool) -> None:
    value = plan(tmp_path)
    original = os.replace
    interrupted = False

    def interrupt(source: Path, target: Path) -> None:
        nonlocal interrupted
        pending = json.loads(source.read_bytes())
        if pending["completed"] and not interrupted:
            interrupted = True
            if after:
                original(source, target)
            raise KeyboardInterrupt()
        original(source, target)

    monkeypatch.setattr(os, "replace", interrupt)
    assert_error("CANCELLED", lambda: execute(value))
    state_path = receipt(tmp_path)
    state = hf._read_state(state_path)
    assert state["stage"] == "interrupted" and len(state["completed"]) == 1
    assert not (state_path.parent / "receipt.pending").exists()
    completed = {item["path"] for item in state["completed"]}
    remote.downloads.clear()
    execute(plan(tmp_path, resume=state_path), resume=state_path)
    assert not completed & set(remote.downloads)
    assert hf._read_state(state_path)["stage"] == "completed"


def test_interruption_discarding_verified_partial_can_resume(tmp_path: Path, remote: Transport,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    value = plan(tmp_path)
    remote.interrupt_path = "model_ctx.onnx"
    assert_error("CANCELLED", lambda: execute(value))
    state_path = receipt(tmp_path)
    remote.interrupt_path = None
    original = Path.unlink
    interrupted = False

    def interrupt(path: Path, missing_ok: bool = False) -> None:
        nonlocal interrupted
        original(path, missing_ok=missing_ok)
        if path == state_path.parent / "partial" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(Path, "unlink", interrupt)
    assert_error("CANCELLED", lambda: execute(plan(tmp_path, resume=state_path), resume=state_path))
    assert hf._read_state(state_path)["partial"] is None
    execute(plan(tmp_path, resume=state_path), resume=state_path)
    assert hf._read_state(state_path)["stage"] == "completed"


def test_promotion_interruption_resume(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    value = plan(tmp_path)
    original = Path.rename

    def interrupt(source: Path, target: Path) -> Path:
        result = original(source, target)
        if source.name == "bundle":
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(Path, "rename", interrupt)
    assert_error("CANCELLED", lambda: execute(value))
    state_path = receipt(tmp_path)
    assert hf._read_state(state_path)["stage"] == "promoting"
    assert Path(value.destination).is_dir() and not (state_path.parent / "bundle").exists()
    before = len(remote.downloads)
    execute(plan(tmp_path, resume=state_path), resume=state_path)
    assert len(remote.downloads) == before
    assert hf._read_state(state_path)["stage"] == "completed"


@pytest.mark.parametrize("empty", [False, True])
def test_promotion_race_never_replaces_destination(tmp_path: Path, remote: Transport,
                                                  monkeypatch: pytest.MonkeyPatch, empty: bool) -> None:
    value = plan(tmp_path)
    destination = Path(value.destination)
    original = Path.rename

    def race(source: Path, target: Path) -> Path:
        if source.name == "bundle":
            destination.mkdir()
            if not empty:
                (destination / "user.txt").write_text("preserve")
        return original(source, target)

    monkeypatch.setattr(Path, "rename", race)
    assert_error("LOCAL_IO_FAILED", lambda: execute(value))
    state_path = receipt(tmp_path)
    assert (state_path.parent / "bundle").is_dir()
    assert list(destination.iterdir()) == ([] if empty else [destination / "user.txt"])
    if not empty:
        assert (destination / "user.txt").read_text() == "preserve"
    assert_error("DESTINATION_EXISTS", lambda: execute(plan(tmp_path, resume=state_path), resume=state_path))


def test_revalidation_preserves_total_timeout_budget(tmp_path: Path, remote: Transport,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    value = plan(tmp_path)
    original = hf.plan_acquisition
    budgets: list[float] = []

    def record(*args: Any, **kwargs: Any) -> hf.AcquisitionPlan:
        budgets.append(kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(hf, "plan_acquisition", record)
    execute(value, timeout=300)
    assert len(budgets) == 1 and 30 < budgets[0] <= 300


@pytest.mark.parametrize("tamper", ["receipt", "staged", "partial", "extra", "destination", "options", "metadata"])
def test_resume_fails_closed(tmp_path: Path, remote: Transport, tamper: str) -> None:
    value = plan(tmp_path)
    remote.interrupt_path = "model_ctx.onnx"
    assert_error("CANCELLED", lambda: execute(value))
    path = receipt(tmp_path)
    state = hf._read_state(path)
    code = "STAGED_TAMPER"
    if tamper == "receipt":
        text = path.read_text().replace('"interrupted"', '"failed"')
        path.write_text(text)
        code = "RECEIPT_TAMPER"
    elif tamper == "staged":
        (path.parent / "bundle" / state["completed"][0]["path"]).write_bytes(b"changed")
        code = "CHECKSUM_MISMATCH"
    elif tamper == "partial":
        (path.parent / "partial").write_bytes(b"changed")
    elif tamper == "extra":
        (path.parent / "bundle" / "extra.txt").write_text("unexpected")
    elif tamper == "destination":
        Path(value.destination).mkdir()
        (Path(value.destination) / "user.txt").write_text("preserve")
        code = "DESTINATION_EXISTS"
    elif tamper == "options":
        code = "PLAN_CHANGED"
    elif tamper == "metadata":
        remote.metadata["siblings"][0]["size"] += 1
        code = "METADATA_DRIFT"
    before = path.read_bytes()
    remote.downloads.clear()
    remote.interrupt_path = None

    def attempt() -> None:
        current = plan(tmp_path, resume=path, processor_name="changed" if tamper == "options" else None)
        execute(current, resume=path)

    assert_error(code, attempt)
    assert path.read_bytes() == before and not remote.downloads
    if tamper == "destination":
        assert (Path(value.destination) / "user.txt").read_text() == "preserve"


def test_fingerprint_consent_and_overwrite(tmp_path: Path, remote: Transport) -> None:
    value = plan(tmp_path)
    assert_error("PLAN_CHANGED", lambda: hf.execute_acquisition(value, expected_fingerprint="b" * 64,
                                                               accept_license=True))
    assert_error("RIGHTS_CONSENT_REQUIRED", lambda: hf.execute_acquisition(
        value, expected_fingerprint=value.fingerprint, accept_license=False))
    assert not remote.downloads and not list(tmp_path.glob("hf-acquire-*"))
    Path(value.destination).mkdir()
    marker = Path(value.destination) / "user.txt"
    marker.write_text("preserved")
    assert_error("DESTINATION_EXISTS", lambda: execute(value))
    assert marker.read_text() == "preserved"


@pytest.mark.parametrize("name", ["context.qnn", "genai_config.json"])
def test_lfs_and_git_checksums(tmp_path: Path, remote: Transport, name: str) -> None:
    value = plan(tmp_path)
    remote.corrupt = name
    assert_error("CHECKSUM_MISMATCH", lambda: execute(value))
    state = hf._read_state(receipt(tmp_path))
    assert state["stage"] == "failed" and state["error"]["code"] == "CHECKSUM_MISMATCH"
    assert not Path(value.destination).exists()
    remote.corrupt = None
    execute(plan(tmp_path, resume=receipt(tmp_path)), resume=receipt(tmp_path))


@pytest.mark.parametrize("drift", ["during", "length", "before"])
def test_metadata_drift(tmp_path: Path, remote: Transport, drift: str) -> None:
    value = plan(tmp_path)
    if drift == "during":
        remote.drift_after_download = True
    elif drift == "length":
        remote.length_override = "123456"
    else:
        remote.metadata["siblings"][0]["size"] += 1
    assert_error("METADATA_DRIFT", lambda: execute(value))
    assert not Path(value.destination).exists()
    if drift == "before":
        assert not remote.downloads and not list(tmp_path.glob("hf-acquire-*"))
    else:
        assert hf._read_state(receipt(tmp_path))["stage"] == "failed"


def test_structural_failure_not_promoted(tmp_path: Path, remote: Transport) -> None:
    remote.data["genai_config.json"] = b"{}"
    remote.refresh()
    value = plan(tmp_path)
    assert_error("STRUCTURAL_MISMATCH", lambda: execute(value))
    assert not Path(value.destination).exists()
    assert hf._read_state(receipt(tmp_path))["stage"] == "failed"


@pytest.mark.parametrize("field,value", [("private", True), ("gated", "auto"), ("gated", None)])
def test_public_only(tmp_path: Path, remote: Transport, field: str, value: Any) -> None:
    remote.metadata[field] = value
    assert_error("HF_PUBLIC_ONLY", lambda: plan(tmp_path))
    assert not remote.downloads


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_errors_no_retry(tmp_path: Path, remote: Transport, status: int) -> None:
    remote.metadata_status = status
    assert_error("HF_PUBLIC_ONLY", lambda: plan(tmp_path))
    assert len(remote.requests) == 1


def test_ambient_auth_endpoint_ignored(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT", "HTTPS_PROXY", "HTTP_PROXY"):
        monkeypatch.setenv(name, "https://do-not-contact.invalid/secret")
    execute(plan(tmp_path))
    assert {host for host, _, _ in remote.requests} == {"huggingface.co"}
    assert all("secret" not in path and "authorization" not in {key.lower() for key in headers}
               for _, path, headers in remote.requests)


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://evil.invalid/file",
                               "https://huggingface.co.evil.invalid/file", "https://user@huggingface.co/file",
                               "https://huggingface.co:443/file"])
def test_unsafe_redirects(tmp_path: Path, remote: Transport, url: str) -> None:
    value = plan(tmp_path)
    remote.redirect = url
    assert_error("UNSAFE_REDIRECT", lambda: execute(value))
    assert not Path(value.destination).exists()
    assert {host for host, _, _ in remote.requests} == {"huggingface.co"}


@pytest.mark.parametrize("kwargs,code", [
    ({"repo_id": "https://huggingface.co/x/y"}, "INVALID_REPO"),
    ({"repo_id": "../private"}, "INVALID_REPO"), ({"repo_id": "x/y@secret"}, "INVALID_REPO"),
    ({"revision": "../main"}, "INVALID_REVISION"), ({"subfolder": "../outside"}, "UNSAFE_REFERENCE"),
    ({"subfolder": r"bad\path"}, "UNSAFE_REFERENCE"), ({"subfolder": "CON"}, "UNSAFE_REFERENCE"),
])
def test_unsafe_inputs(tmp_path: Path, remote: Transport, kwargs: dict[str, Any], code: str) -> None:
    options = {"repo_id": REPO, **kwargs}
    assert_error(code, lambda: hf.plan_acquisition(destination=tmp_path / "out", work_dir=tmp_path, **options))
    assert not remote.requests


@pytest.mark.parametrize("name,code", [
    ("../outside.bin", "UNSAFE_REFERENCE"), ("CON.bin", "UNSAFE_REFERENCE"),
    ("bad:.bin", "UNSAFE_REFERENCE"), ("a//file.bin", "UNSAFE_REFERENCE"),
    ("TOKENIZER.json", "PATH_ALIAS"), ("sub/file.txt", "PATH_ALIAS"),
    ("model.pyc", "UNSUPPORTED_FILE"), ("README.exe", "UNSUPPORTED_FILE"),
    ("model.pyd", "UNSUPPORTED_FILE"), ("model.dll", "UNSUPPORTED_FILE"),
    ("script.ps1", "UNSUPPORTED_FILE"), ("script.bat", "UNSUPPORTED_FILE"),
    ("script.sh", "UNSUPPORTED_FILE"), ("weights.pt", "UNSUPPORTED_FILE"),
    ("model.safetensors", "UNSUPPORTED_FILE"), ("archive.zip", "UNSUPPORTED_FILE"),
    ("archive.tar.gz", "UNSUPPORTED_FILE"),
    ("pytorch_model.bin", "UNSUPPORTED_FILE"), (".cache/file.bin", "UNSUPPORTED_FILE"),
    (".GIT/file.bin", "UNSUPPORTED_FILE"), (".CACHE/file.bin", "UNSUPPORTED_FILE"),
    ("bad\ud800.txt", "INVALID_METADATA"),
])
def test_inventory_rejected(tmp_path: Path, remote: Transport, name: str, code: str) -> None:
    remote.data[name] = b"invalid"
    if name == "sub/file.txt":
        remote.data["Sub/other.txt"] = b"case-directory-alias"
    remote.refresh()
    assert_error(code, lambda: plan(tmp_path))
    assert not remote.downloads


@pytest.mark.parametrize("change,code", [
    ("empty", "INVENTORY_LIMIT"), ("count", "INVENTORY_LIMIT"), ("size", "INVENTORY_LIMIT"),
    ("total", "INVENTORY_LIMIT"), ("lfs", "INVALID_METADATA"), ("git", "INVALID_METADATA"),
    ("missing", "MISSING_ASSET"), ("duplicate", "PATH_ALIAS"),
])
def test_inventory_bounds(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch,
                          change: str, code: str) -> None:
    siblings = remote.metadata["siblings"]
    if change == "empty":
        remote.metadata["siblings"] = []
    elif change == "count":
        remote.metadata["siblings"] *= 1000
    elif change == "size":
        siblings[0]["size"] = hf.MAX_FILE + 1
    elif change == "total":
        monkeypatch.setattr(hf, "MAX_TOTAL", 1)
    elif change == "lfs":
        next(item for item in siblings if "lfs" in item)["lfs"]["sha256"] = "a" * 40
    elif change == "git":
        siblings[0]["blobId"] = "a" * 64
    elif change == "missing":
        remote.metadata["siblings"] = [item for item in siblings if item["rfilename"] != "tokenizer.json"]
    elif change == "duplicate":
        siblings.append(copy.deepcopy(siblings[0]))
    assert_error(code, lambda: plan(tmp_path))
    assert not remote.downloads


def test_cli_plan_and_resume_boundary(tmp_path: Path, remote: Transport) -> None:
    args = ["acquire-prebuilt", "--repo", REPO, "--out", str(tmp_path / "acquired"),
            "--work-dir", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["executed"] is False and not remote.downloads
    rejected = CliRunner().invoke(app, args + ["--resume", str(tmp_path / "receipt.json")])
    assert rejected.exit_code == 2 and "RESUME_REQUIRES_EXECUTE" in rejected.output
    result = CliRunner().invoke(app, args + ["--execute", "--expect-fingerprint", value["fingerprint"],
                                            "--accept-license"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["stage"] == "completed"


def test_reparse_work_parent(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.lstat

    def reparse(path: Path, **kwargs: Any) -> Any:
        info = original(path, **kwargs)
        if path == tmp_path:
            from types import SimpleNamespace
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", reparse)
    assert_error("SYMLINK_ASSET", lambda: plan(tmp_path))
    assert not remote.requests


def test_real_onnx_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle: Path) -> None:
    onnx = pytest.importorskip("onnx")
    node = onnx.helper.make_node("EPContext", [], ["context"], domain="com.microsoft",
                                source="QNN", embed_mode=0, main_context=1, ep_cache_context="context.qnn")
    graph = onnx.helper.make_graph([node], "context_fixture", [],
                                  [onnx.helper.make_tensor_value_info("context", onnx.TensorProto.FLOAT, [1])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.save(model, str(bundle / "model_ctx.onnx"))
    remote = Transport(monkeypatch, bundle)
    result = execute(plan(tmp_path))
    assert remote.downloads and result["inspection"]["levels"]["structural"]["status"] == "passed"
    assert result["inspection"]["levels"]["native_evidence"]["status"] == "not-run"


def test_atomic_state_failure(tmp_path: Path, remote: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    value = plan(tmp_path)
    remote.interrupt_path = "model_ctx.onnx"
    assert_error("CANCELLED", lambda: execute(value))
    path = receipt(tmp_path)
    before = path.read_bytes()
    state = hf._read_state(path)
    original = os.replace

    def fail_replace(source: Path, destination: Path) -> None:
        if destination == path:
            raise OSError("synthetic state replace failure")
        original(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    state["stage"] = "failed"
    with pytest.raises(OSError):
        hf._save(path, state)
    assert path.read_bytes() == before
    assert hf._read_state(path)["stage"] == "interrupted"
    assert not (path.parent / "receipt.pending").exists()
    monkeypatch.setattr(os, "replace", original)
    remote.interrupt_path = None
    execute(plan(tmp_path, resume=path), resume=path)
    assert hf._read_state(path)["stage"] == "completed"
