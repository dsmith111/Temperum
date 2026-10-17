from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.host import HostInfo
from npu_model.profiles import get_runtime
from npu_model.toolchains import LOCKS, STATE_FILE, Toolchains, parse_lock, platform_tag, python_request

RUNTIME = get_runtime("fork-qnn-1.24.1")
ARM64 = HostInfo(system="Windows", native_arch="arm64", python_platform="win-amd64",
                 processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU")
X64 = HostInfo(system="Windows", native_arch="x64", python_platform="win-amd64", processor_name="Intel",
               cuda_gpus=["NVIDIA RTX"])


class FakeSystem:
    """Simulates uv, venv, pip and the environment probe."""

    def __init__(self, root: Path, platform: str = "win-arm64", pip_failures: int = 0,
                 probe: dict[str, Any] | None = None) -> None:
        self.root = root
        self.platform = platform
        self.pip_failures = pip_failures
        self.probe = probe
        self.installed: list[dict[str, str]] = []
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        out = ""
        if cmd[:3] == ["uv", "python", "list"]:
            out = json.dumps(self.installed)
        elif cmd[:3] == ["uv", "python", "install"]:
            arch = "aarch64" if self.platform == "win-arm64" else "x86_64"
            exe = self.root / "python" / f"cpython-3.12.15-windows-{arch}-none" / "python.exe"
            exe.parent.mkdir(parents=True)
            exe.write_text("", encoding="utf-8")
            self.installed.append({"key": f"cpython-3.12.15-windows-{arch}-none", "path": str(exe)})
        elif cmd[1:3] == ["-m", "venv"]:
            exe = Path(cmd[3]) / "Scripts" / "python.exe"
            exe.parent.mkdir(parents=True)
            exe.write_text("", encoding="utf-8")
        elif cmd[1:4] == ["-m", "pip", "install"]:
            if self.pip_failures:
                self.pip_failures -= 1
                return subprocess.CompletedProcess(cmd, 1, "", "ERROR: HTTP error 404 while getting wheel")
        elif cmd[1] == "-c" and "importlib" in cmd[2]:
            out = json.dumps(self.probe or self.default_probe(cmd[2]))
        elif cmd[1] == "-c":
            out = f"3.12 {self.platform}"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    def default_probe(self, script: str) -> dict[str, Any]:
        versions = {"onnxruntime": "1.24.1", "onnxruntime-genai": "0.12.2"}
        return {"platform": self.platform, "python": "3.12", "versions": versions,
                "providers": ["QNNExecutionProvider", "CPUExecutionProvider"], "cuda": False, "error": None}

    def pip_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:4] == ["-m", "pip", "install"]]


@pytest.fixture(autouse=True)
def _long_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("npu_model.toolchains.long_paths_enabled", lambda: True)  # pytest temp dirs are deep


def _toolchains(tmp_path: Path, system: FakeSystem, host: HostInfo = ARM64, device: str = "cpu",
                **kwargs: Any) -> Toolchains:
    return Toolchains(host, RUNTIME, device, home=tmp_path, run=system, uv="uv", sleep=lambda _s: None, **kwargs)


def test_shipped_locks_are_consistent() -> None:
    locks = sorted((LOCKS / RUNTIME.id).glob("*.txt"))
    assert locks
    for path in locks:
        lock = parse_lock(path)
        assert path.stem.startswith(f"{lock.role}-{lock.platform}")
        names = {p.split("==")[0].lower() for p in lock.pins}
        assert all("==" in p for p in lock.pins), path.name
        if lock.role in ("compile", "verify"):
            assert "onnxruntime-qnn==1.24.1" in lock.pins and "onnxruntime" not in names, path.name
        if lock.role == "verify":
            assert "onnxruntime-genai==0.12.2" in lock.pins
        if lock.role == "quantize":
            assert {"olive-ai", "transformers", "torch", "onnxruntime-genai", "accelerate", "psutil"} <= names
        if lock.platform == "win-arm64":
            assert lock.mode == "exact" and lock.tested.startswith("2026-10-17"), path.name


def test_python_and_platform_names() -> None:
    assert platform_tag(ARM64) == "win-arm64" and platform_tag(X64) == "win-amd64"
    assert python_request(ARM64, "3.12") == "cpython-3.12-windows-aarch64-none"


def test_build_uses_native_python_and_pinned_packages(tmp_path: Path) -> None:
    system = FakeSystem(tmp_path)
    steps: list[str] = []
    chains = _toolchains(tmp_path, system, progress=steps.append)
    assert chains.status("quantize").state == "missing"
    result = chains.ensure("quantize")
    assert result.state == "ready"
    assert ["uv", "python", "install", "cpython-3.12-windows-aarch64-none"] in system.calls
    torch_call, rest_call = system.pip_calls()
    assert "--no-deps" in torch_call and torch_call[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert any(arg.startswith("torch==2.14.1") for arg in torch_call)
    assert "--no-deps" in rest_call and "-r" in rest_call
    requirements = Path(rest_call[rest_call.index("-r") + 1]).read_text(encoding="utf-8")
    assert "olive-ai==0.13.0" in requirements and "transformers==4.57.3" in requirements
    state = json.loads((chains.env_path("quantize") / STATE_FILE).read_text(encoding="utf-8"))
    assert state["lock"] == "quantize-win-arm64-cpu.txt" and state["platform"] == "win-arm64"
    assert any("download.pytorch.org" in s for s in steps)

    calls = len(system.calls)
    assert chains.ensure("quantize").state == "ready" and len(system.calls) == calls  # no rebuild


def test_changed_lock_marks_the_env_stale(tmp_path: Path) -> None:
    system = FakeSystem(tmp_path)
    chains = _toolchains(tmp_path, system)
    chains.ensure("verify")
    state_file = chains.env_path("verify") / STATE_FILE
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state_file.write_text(json.dumps({**state, "lock_sha256": "0" * 64}), encoding="utf-8")
    assert chains.status("verify").state == "stale"
    assert chains.ensure("verify").state == "ready"


def test_pip_is_retried_when_the_feed_hiccups(tmp_path: Path) -> None:
    system = FakeSystem(tmp_path, pip_failures=2)
    chains = _toolchains(tmp_path, system)
    assert chains.ensure("verify").state == "ready"
    assert len(system.pip_calls()) == 3
    assert "404" in chains.log_path("verify").read_text(encoding="utf-8")


def test_pip_failure_points_to_the_log(tmp_path: Path) -> None:
    chains = _toolchains(tmp_path, FakeSystem(tmp_path, pip_failures=99))
    with pytest.raises(NpuModelError) as exc:
        chains.ensure("verify")
    assert exc.value.reason_code == "PIP_FAILED" and "setup-fork-qnn-1.24.1-verify.log" in (exc.value.hint or "")


def test_environment_with_the_wrong_runtime_fails_its_check(tmp_path: Path) -> None:
    probe = {"platform": "win-arm64", "versions": {"onnxruntime": "1.31.0"}, "providers": ["CPUExecutionProvider"],
             "error": None}
    chains = _toolchains(tmp_path, FakeSystem(tmp_path, probe=probe))
    with pytest.raises(NpuModelError) as exc:
        chains.ensure("compile")
    assert exc.value.reason_code == "ENV_BROKEN"
    assert "1.31.0" in exc.value.message and "QNNExecutionProvider" in exc.value.message


def test_emulated_python_is_rejected(tmp_path: Path) -> None:
    python = tmp_path / "py" / "python.exe"
    python.parent.mkdir()
    python.write_text("", encoding="utf-8")
    chains = _toolchains(tmp_path, FakeSystem(tmp_path, platform="win-amd64"), python=python)
    with pytest.raises(NpuModelError) as exc:
        chains.ensure("verify")
    assert exc.value.reason_code == "WRONG_PYTHON"


def test_cuda_resolve_lock_installs_genai_without_dependencies(tmp_path: Path) -> None:
    probe = {"platform": "win-amd64", "versions": {"onnxruntime": "1.24.4", "onnxruntime-genai": "0.12.2"},
             "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda": True, "error": None}
    system = FakeSystem(tmp_path, platform="win-amd64", probe=probe)
    chains = _toolchains(tmp_path, system, host=X64, device="cuda")
    assert chains.ensure("quantize").state == "ready"
    torch_call, rest_call, genai_call = system.pip_calls()
    assert torch_call[-1] == "https://download.pytorch.org/whl/cu126"
    assert "--no-deps" not in rest_call
    assert "--no-deps" in genai_call and "onnxruntime-genai==0.12.2" in genai_call


def test_unsupported_platform_has_no_toolchain(tmp_path: Path) -> None:
    linux = HostInfo(system="Linux", native_arch="x64", python_platform="linux-x86_64", processor_name=None)
    with pytest.raises(NpuModelError) as exc:
        _toolchains(tmp_path, FakeSystem(tmp_path), host=linux).status("compile")
    assert exc.value.reason_code == "NO_TOOLCHAIN"


def test_deep_env_roots_are_refused_without_long_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("npu_model.toolchains.long_paths_enabled", lambda: False)
    deep = tmp_path / ("x" * 120)
    with pytest.raises(NpuModelError) as exc:
        _toolchains(deep, FakeSystem(deep)).ensure("verify")
    assert exc.value.reason_code == "PATH_TOO_LONG"
