"""Facts about the machine the converter runs on: architecture, CPU/NPU identity, memory, CUDA.

The default Python on Windows-on-Arm laptops is often the x64 build under emulation, so the
native architecture comes from the system-wide PROCESSOR_ARCHITECTURE, not the interpreter.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from npu_model.core.errors import NpuModelError
from npu_model.profiles import TARGETS, TargetProfile, get_target

Runner = Callable[..., "subprocess.CompletedProcess[str]"]



@dataclass
class HostInfo:
    system: str
    native_arch: str
    python_platform: str
    processor_name: str | None
    npu_devices: list[str] = field(default_factory=list)
    cuda_gpus: list[str] = field(default_factory=list)
    total_ram_gb: float | None = None
    available_ram_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ARCH_NAMES = {"arm64": "arm64", "aarch64": "arm64", "amd64": "x64", "x86_64": "x64", "x86": "x86"}


def native_arch() -> str:
    """Native machine architecture, even from an emulated (x64-on-ARM64) interpreter.

    An emulated x64 process sees PROCESSOR_ARCHITECTURE=AMD64 and IsWow64Process2 does not
    help (it covers WOW64, not x64 emulation), so on Windows read the system-wide value.
    """
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            value = str(winreg.QueryValueEx(key, "PROCESSOR_ARCHITECTURE")[0])
            if value.lower() in _ARCH_NAMES:
                return _ARCH_NAMES[value.lower()]
        except OSError:
            pass
    machine = platform.machine().lower()  # CPython >= 3.12 reports the native machine on Windows
    if machine in _ARCH_NAMES:
        return _ARCH_NAMES[machine]
    env = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    return _ARCH_NAMES.get(env.lower(), env.lower() or machine)


def _processor_name() -> str | None:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or None
        except OSError:
            return None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Hardware", "Processor") and value.strip():
                    return value.strip()
    except OSError:
        pass
    return platform.processor() or None


def _npu_devices(run: Runner) -> list[str]:
    if sys.platform != "win32":
        return []
    script = ("Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | "
              "Where-Object { $_.Class -eq 'ComputeAccelerator' -or $_.FriendlyName -match 'NPU' } | "
              "Select-Object -ExpandProperty FriendlyName | ConvertTo-Json -Compress")
    try:
        proc = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return []
    try:
        names = json.loads(out)
    except ValueError:
        return []
    names = [names] if isinstance(names, str) else names
    return sorted({str(n) for n in names if n})


def _cuda_gpus(run: Runner) -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()] if proc.returncode == 0 else []


def memory_gb() -> tuple[float | None, float | None]:
    """(total, available) physical memory in GB."""
    if sys.platform == "win32":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
        return None, None
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                info[key] = int(value.split()[0]) * 1024
        return info["MemTotal"] / 2**30, info.get("MemAvailable", info["MemFree"]) / 2**30
    except (OSError, KeyError, ValueError):
        return None, None


def long_paths_enabled() -> bool:
    """Whether Windows allows paths beyond 260 characters (always true elsewhere)."""
    if sys.platform != "win32":
        return True
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        return int(winreg.QueryValueEx(key, "LongPathsEnabled")[0]) == 1
    except (OSError, ValueError):
        return False


def detect_host(run: Runner = subprocess.run) -> HostInfo:
    total, available = memory_gb()
    return HostInfo(
        system=platform.system(),
        native_arch=native_arch(),
        python_platform=sysconfig.get_platform(),
        processor_name=_processor_name(),
        npu_devices=_npu_devices(run),
        cuda_gpus=_cuda_gpus(run),
        total_ram_gb=round(total, 1) if total else None,
        available_ram_gb=round(available, 1) if available else None,
    )


def resolve_target(host: HostInfo, requested: str | None = None) -> TargetProfile:
    """Pick the target profile: an explicit id, or the one matching this machine's processor."""
    if requested and requested != "auto":
        return get_target(requested)
    matches = [t for t in TARGETS.values() if t.matches(host.processor_name)]
    if not matches:
        raise NpuModelError(
            "target", "NO_MATCHING_TARGET",
            f"No NPU target profile matches this machine ({host.processor_name or 'unknown processor'}).",
            hint="Pass --target explicitly to build for another device "
                 f"(known: {', '.join(sorted(TARGETS))}); the bundle stays unverified until run there.")
    return matches[0]
