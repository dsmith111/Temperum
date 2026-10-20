"""Isolated Python environments ("roles") that run the heavy toolchains.

  quantize  torch, transformers, Olive, ONNX Runtime (CPU or CUDA) and the GenAI model
            builder: rotation, weight quantization, ONNX export, activation calibration,
            and the fp32 reference used by verification.
  compile   Olive with the onnxruntime-qnn build whose QNN SDK matches the runtime: compiles
            the NPU context binaries (contexts from a newer QNN generally fail to load on an
            older runtime).
  verify    onnxruntime-qnn and onnxruntime-genai as the consumer runtime has them: loads and
            runs the bundle on the NPU.

Each role is built from a committed lock file, locks/<runtime>/<role>-<platform>[-<device>].txt,
using a uv-managed CPython for the machine's native architecture (the default Python on
Windows on Arm is often an emulated x64 build). Packages come through the user's own pip
configuration (index, proxy, certificates). Only torch comes from the official PyTorch
index, because PyPI has no Windows ARM64 or Windows CUDA builds of it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from temperum import paths
from temperum.core.errors import TemperumError
from temperum.host import HostInfo, long_paths_enabled
from temperum.profiles import RuntimeProfile

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
Progress = Callable[[str], None]

ROLES = ("quantize", "compile", "verify")
LOCKS = Path(__file__).resolve().parent / "locks"
STATE_FILE = "temperum-env.json"
PIP_FLAGS = ["--disable-pip-version-check", "--no-input", "--progress-bar", "off"]
PIP_ATTEMPTS = 3  # feeds that proxy PyPI on demand can 404 a wheel on its first request
MAX_PATH = 259
DEEPEST_ENV_FILE = 150  # longest path inside the M0 environments, relative to the env root: 144
ENV_SIZE_GB = {"quantize": 1.0, "compile": 1.0, "verify": 0.35}
PYTHON_SIZE_GB = 0.06

# (import name, distribution name) that must import in each role.
ROLE_MODULES: dict[str, list[tuple[str, str | None]]] = {
    "quantize": [("torch", "torch"), ("transformers", "transformers"), ("olive", "olive-ai"),
                 ("onnxruntime", None), ("onnxruntime_genai", "onnxruntime-genai"),
                 ("accelerate", "accelerate"), ("psutil", "psutil"), ("requests", "requests"),
                 ("sentencepiece", "sentencepiece")],  # llama.cpp's GGUF converter imports it for every model
    "compile": [("olive", "olive-ai"), ("onnxruntime", None), ("requests", "requests")],
    "verify": [("onnxruntime", None), ("onnxruntime_genai", "onnxruntime-genai"), ("numpy", "numpy")],
}

_PROBE = r"""
import importlib, importlib.metadata as md, json, sys, sysconfig
out = {"platform": sysconfig.get_platform(), "python": "%d.%d" % sys.version_info[:2],
       "versions": {}, "providers": [], "cuda": None, "error": None}
try:
    for module, dist in MODULES:
        mod = importlib.import_module(module)
        out["versions"][dist or module] = md.version(dist) if dist else getattr(mod, "__version__", None)
    if "onnxruntime" in out["versions"]:
        import onnxruntime
        out["providers"] = onnxruntime.get_available_providers()
    if "torch" in out["versions"]:
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


def _dist_name(pin: str) -> str:
    return re.split(r"[=<>!~\[; ]", pin, maxsplit=1)[0].strip().lower().replace("_", "-")


@dataclass(frozen=True)
class Lock:
    path: Path
    role: str
    platform: str
    device: str | None
    mode: str  # "exact": install every pin with --no-deps; "resolve": let pip resolve the rest
    python: str
    torch_index: str | None
    no_deps: frozenset[str]
    pins: tuple[str, ...]
    tested: str

    @property
    def sha256(self) -> str:
        text = self.path.read_bytes().replace(b"\r\n", b"\n")  # stable across checkout line endings
        return hashlib.sha256(text).hexdigest()

    @property
    def torch_pin(self) -> str | None:
        return next((p for p in self.pins if _dist_name(p) == "torch"), None)


def parse_lock(path: Path) -> Lock:
    fields: dict[str, str] = {}
    pins: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("# temperum toolchain lock:", "# npu-model toolchain lock:")):  # older locks
            for item in line.split(":", 1)[1].split():
                key, _, value = item.partition("=")
                fields[key] = value
        elif line.startswith("#"):
            match = re.match(r"^#\s*([a-z-]+):\s*(.*)$", line)
            if match:
                fields[match.group(1)] = match.group(2).strip()
        else:
            pins.append(line)
    try:
        mode = fields["mode"]
        if mode not in ("exact", "resolve"):
            raise ValueError(mode)
        return Lock(path=path, role=fields["role"], platform=fields["platform"], device=fields.get("device"),
                    mode=mode, python=fields["python"], torch_index=fields.get("torch-index") or None,
                    no_deps=frozenset(_dist_name(n) for n in fields.get("no-deps", "").split(",") if n.strip()),
                    pins=tuple(pins), tested=fields.get("tested", "no"))
    except (KeyError, ValueError) as exc:
        raise TemperumError("toolchain", "BAD_LOCK", f"Malformed toolchain lock {path.name}: {exc}") from exc


def platform_tag(host: HostInfo) -> str:
    """sysconfig-style platform of a native interpreter on this host, e.g. win-arm64."""
    system = host.system.lower()
    if system == "windows":
        return {"arm64": "win-arm64", "x64": "win-amd64"}.get(host.native_arch, f"win-{host.native_arch}")
    return f"{system}-{ {'arm64': 'aarch64', 'x64': 'x86_64'}.get(host.native_arch, host.native_arch) }"


def python_request(host: HostInfo, version: str) -> str:
    """uv's request string for a native CPython, e.g. cpython-3.12-windows-aarch64-none."""
    os_name = {"windows": "windows", "darwin": "macos"}.get(host.system.lower(), host.system.lower())
    arch = {"arm64": "aarch64", "x64": "x86_64"}.get(host.native_arch, host.native_arch)
    return f"cpython-{version}-{os_name}-{arch}-{'gnu' if os_name == 'linux' else 'none'}"


def find_lock(runtime_id: str, role: str, platform: str, device: str | None = None) -> Lock:
    name = f"{role}-{platform}-{device}.txt" if role == "quantize" else f"{role}-{platform}.txt"
    path = LOCKS / runtime_id / name
    if not path.is_file():
        available = sorted(p.stem for p in (LOCKS / runtime_id).glob(f"{role}-*.txt"))
        where = f"{platform} ({device})" if role == "quantize" else platform
        raise TemperumError("toolchain", "NO_TOOLCHAIN",
                            f"No {role} toolchain for {where} with runtime {runtime_id}.",
                            hint=f"Available {role} toolchains: {', '.join(available) or 'none'}.")
    return parse_lock(path)


@dataclass
class EnvStatus:
    role: str
    path: Path
    state: str  # "ready" | "missing" | "stale" | "broken"
    detail: str = ""
    probe: dict[str, Any] | None = None
    tested: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "path": str(self.path), "state": self.state, "detail": self.detail,
                "probe": self.probe, "tested": self.tested}


def _host_of(url: str) -> str:
    return urlsplit(url).hostname or url


class Toolchains:
    """Provision and check the role environments for one runtime on this host."""

    def __init__(self, host: HostInfo, runtime: RuntimeProfile, device: str = "cpu", *,
                 home: Path | None = None, run: Runner = subprocess.run, uv: str | None = None,
                 python: Path | None = None, progress: Progress | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.host = host
        self.runtime = runtime
        self.device = device
        self.home = home or paths.home()
        self.platform = platform_tag(host)
        self._run_fn = run
        self._uv = uv
        self._python_override = python
        self._progress = progress or (lambda _msg: None)
        self._sleep = sleep

    # -- layout ---------------------------------------------------------------------------
    def env_path(self, role: str) -> Path:
        return self.home / "envs" / self.runtime.id / role

    def python_exe(self, role: str) -> Path:
        env = self.env_path(role)
        return env / "Scripts" / "python.exe" if self.host.system.lower() == "windows" else env / "bin" / "python"

    def scripts_dir(self, role: str) -> Path:
        return self.python_exe(role).parent

    def lock(self, role: str) -> Lock:
        return find_lock(self.runtime.id, role, self.platform, self.device if role == "quantize" else None)

    def log_path(self, role: str) -> Path:
        return self.home / "logs" / f"setup-{self.runtime.id}-{role}.log"

    # -- status ---------------------------------------------------------------------------
    def status(self, role: str, deep: bool = False) -> EnvStatus:
        lock = self.lock(role)
        path = self.env_path(role)
        if not self.python_exe(role).is_file():
            return EnvStatus(role, path, "missing", tested=lock.tested)
        try:
            state = json.loads((path / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return EnvStatus(role, path, "stale", "setup did not finish", tested=lock.tested)
        if state.get("lock_sha256") != lock.sha256:
            return EnvStatus(role, path, "stale", "toolchain lock changed", tested=lock.tested)
        if not deep:
            return EnvStatus(role, path, "ready", probe=state.get("probe"), tested=lock.tested)
        probe = self._probe(role)
        problems = self._problems(role, lock, probe)
        return EnvStatus(role, path, "broken" if problems else "ready", "; ".join(problems), probe, lock.tested)

    def _problems(self, role: str, lock: Lock, probe: dict[str, Any]) -> list[str]:
        if probe.get("error"):
            return [str(probe["error"])]
        problems: list[str] = []
        if probe.get("platform") != lock.platform:
            problems.append(f"interpreter platform {probe.get('platform')} != {lock.platform}")
        versions = probe.get("versions", {})
        providers = probe.get("providers", [])
        if role in ("compile", "verify"):
            if versions.get("onnxruntime") != self.runtime.ort_version:
                problems.append(f"onnxruntime {versions.get('onnxruntime')} != {self.runtime.ort_version}")
            if "QNNExecutionProvider" not in providers:
                problems.append("QNNExecutionProvider is not available")
        if role == "verify" and versions.get("onnxruntime-genai") != self.runtime.genai_version:
            problems.append(f"onnxruntime-genai {versions.get('onnxruntime-genai')} != {self.runtime.genai_version}")
        if role == "quantize":
            if versions.get("onnxruntime-genai") != self.runtime.builder_genai:
                problems.append(f"onnxruntime-genai {versions.get('onnxruntime-genai')} != {self.runtime.builder_genai}")
            if self.device == "cuda" and not (probe.get("cuda") and "CUDAExecutionProvider" in providers):
                problems.append("CUDA is not usable from the quantize environment")
        return problems

    # -- provisioning ---------------------------------------------------------------------
    def ensure(self, role: str) -> EnvStatus:
        """Build the role environment unless an up-to-date one exists."""
        current = self.status(role)
        if current.state == "ready":
            return current
        return self.build(role)

    def build(self, role: str) -> EnvStatus:
        lock = self.lock(role)
        path = self.env_path(role)
        self._check_path_length(path)
        python = self.base_python(lock.python)
        log = self.log_path(role)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"# {datetime.now(timezone.utc).isoformat()} setup {self.runtime.id}/{role} "
                       f"from {lock.path.name}\n", encoding="utf-8")
        if path.exists():
            self._remove(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._progress(f"{role}: creating the environment")
        self._call([str(python), "-m", "venv", str(path)], log, "VENV_FAILED")
        pip = [str(self.python_exe(role)), "-m", "pip", "install", *PIP_FLAGS]
        torch = lock.torch_pin
        if torch and lock.torch_index:
            self._progress(f"{role}: installing {torch} from {_host_of(lock.torch_index)}")
            self._pip(pip + ["--no-deps", torch, "--index-url", lock.torch_index], log)
        requirements = path / "temperum-requirements.txt"
        requirements.write_text("\n".join(p for p in lock.pins if _dist_name(p) not in lock.no_deps) + "\n",
                                encoding="utf-8")
        count = len(lock.pins)
        self._progress(f"{role}: installing {count} pinned packages through your pip configuration")
        self._pip(pip + (["--no-deps"] if lock.mode == "exact" else []) + ["-r", str(requirements)], log)
        no_deps = [p for p in lock.pins if _dist_name(p) in lock.no_deps]
        if no_deps:
            self._pip(pip + ["--no-deps", *no_deps], log)
        self._progress(f"{role}: checking the environment")
        probe = self._probe(role)
        problems = self._problems(role, lock, probe)
        if problems:
            raise TemperumError("toolchain", "ENV_BROKEN", f"The {role} environment failed its check: "
                                f"{'; '.join(problems)}", hint=f"Setup log: {log}")
        state = {"role": role, "runtime": self.runtime.id, "lock": lock.path.name, "lock_sha256": lock.sha256,
                 "python": str(python), "platform": lock.platform, "device": lock.device, "probe": probe,
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (path / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return EnvStatus(role, path, "ready", probe=probe, tested=lock.tested)

    def base_python(self, version: str) -> Path:
        """A native CPython `version` to create environments from (uv-managed by default)."""
        if self._python_override:
            self._check_interpreter(self._python_override, version)
            return self._python_override
        request = python_request(self.host, version)
        found = self._managed_python(request)
        if found:
            return found
        self._progress(f"Installing Python {version} for {self.platform} (uv: {request})")
        proc = self._run([self._uv_bin(), "python", "install", request], env=self._uv_env())
        if proc.returncode != 0:
            raise TemperumError("toolchain", "PYTHON_UNAVAILABLE", f"uv could not install {request}: "
                                f"{(proc.stderr or proc.stdout or '').strip()[-400:]}",
                                hint="Check network access to GitHub, or pass --python with a native "
                                     f"Python {version} interpreter.")
        found = self._managed_python(request)
        if not found:
            raise TemperumError("toolchain", "PYTHON_UNAVAILABLE", f"uv installed {request} but it was not found.")
        return found

    def _managed_python(self, request: str) -> Path | None:
        # `uv python find` skips interpreters of another architecture when uv itself is an
        # emulated x64 build, so list everything installed and match the key exactly.
        _, version, suffix = request.split("-", 2)
        proc = self._run([self._uv_bin(), "python", "list", "--only-installed", "--all-arches", "--managed-python",
                          "--output-format", "json"], env=self._uv_env())
        try:
            entries = json.loads(proc.stdout or "[]") if proc.returncode == 0 else []
        except ValueError:
            entries = []
        for entry in entries:
            key = str(entry.get("key", ""))
            if key.startswith(f"cpython-{version}.") and key.endswith(f"-{suffix}") and entry.get("path"):
                candidate = Path(str(entry["path"]))
                if candidate.is_file():
                    self._check_interpreter(candidate, version)
                    return candidate
        return None

    def _check_interpreter(self, python: Path, version: str) -> None:
        proc = self._run([str(python), "-c", "import sys, sysconfig; "
                          "print('%d.%d' % sys.version_info[:2], sysconfig.get_platform())"])
        found = (proc.stdout or "").split()
        if proc.returncode != 0 or found != [version, self.platform]:
            raise TemperumError("toolchain", "WRONG_PYTHON",
                                f"{python} is {' '.join(found) or 'not runnable'}; need Python {version} "
                                f"for {self.platform}.",
                                hint="An emulated or different-version interpreter cannot use the locked "
                                     "native packages.")

    def _uv_bin(self) -> str:
        if self._uv:
            return self._uv
        found = shutil.which("uv")
        if not found:
            try:  # the `uv` package (a dependency of temperum) ships the binary
                found = str(importlib.import_module("uv").find_uv_bin())
            except (ImportError, FileNotFoundError):
                found = None
        if not found:
            raise TemperumError("toolchain", "UV_MISSING", "uv is needed to install a native Python.",
                                hint="pip install uv, or pass --python with a native Python interpreter.")
        self._uv = found
        return found

    def _uv_env(self) -> dict[str, str]:
        return {**os.environ, "UV_PYTHON_INSTALL_DIR": str(self.home / "python")}

    def _probe(self, role: str) -> dict[str, Any]:
        script = _PROBE.replace("MODULES", repr(ROLE_MODULES[role]))
        proc = self._run([str(self.python_exe(role)), "-c", script])
        try:
            result = json.loads((proc.stdout or "").strip().splitlines()[-1])
            assert isinstance(result, dict)
            return result
        except (ValueError, IndexError, AssertionError):
            return {"error": (proc.stderr or proc.stdout or "no output").strip()[-400:]}

    def _check_path_length(self, path: Path) -> None:
        if (self.host.system.lower() == "windows" and len(str(path)) + 1 + DEEPEST_ENV_FILE > MAX_PATH
                and not long_paths_enabled()):
            raise TemperumError("toolchain", "PATH_TOO_LONG",
                                f"{path} leaves too little room for package files under Windows' "
                                "260-character path limit.",
                                hint="Set TEMPERUM_HOME to a short folder (for example C:\\npu) or enable "
                                     "Windows long paths.")

    def _remove(self, path: Path) -> None:
        root = (self.home / "envs").resolve()
        if root not in path.resolve().parents:
            raise TemperumError("toolchain", "UNSAFE_DELETE", f"Refusing to delete {path} outside {root}")

        def _retry_writable(func: Callable[..., Any], target: str, _exc: Any) -> None:
            os.chmod(target, stat.S_IWRITE)
            func(target)

        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry_writable)
        else:  # pragma: no cover - Python < 3.12
            shutil.rmtree(path, onerror=_retry_writable)

    # -- subprocess helpers ---------------------------------------------------------------
    def _run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        try:
            return self._run_fn(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
        except OSError as exc:
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))

    def _call(self, cmd: list[str], log: Path, code: str) -> None:
        proc = self._run(cmd)
        self._append(log, cmd, proc)
        if proc.returncode != 0:
            raise TemperumError("toolchain", code, f"Command failed ({proc.returncode}): {' '.join(cmd[:4])} ...",
                                hint=f"Setup log: {log}")

    def _pip(self, cmd: list[str], log: Path) -> None:
        for attempt in range(1, PIP_ATTEMPTS + 1):
            proc = self._run(cmd)
            self._append(log, cmd, proc)
            if proc.returncode == 0:
                return
            if attempt < PIP_ATTEMPTS:
                self._progress(f"pip failed (attempt {attempt}/{PIP_ATTEMPTS}); retrying")
                self._sleep(5.0 * attempt)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise TemperumError("toolchain", "PIP_FAILED", "Package installation failed: " + " | ".join(tail),
                            hint=f"Full log: {log}. Packages come from your pip configuration "
                                 "(index, proxy); check it can reach the pinned versions.")

    @staticmethod
    def _append(log: Path, cmd: list[str], proc: subprocess.CompletedProcess[str]) -> None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(cmd)}\n[exit {proc.returncode}]\n{proc.stdout or ''}{proc.stderr or ''}")
