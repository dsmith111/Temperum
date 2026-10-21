"""Live machine load for the working screen: the CPU as a whole, memory in use, and the processor time of
the fork's own processes, sampled once a second by a single PowerShell process.

Windows exposes no utilization counter for the Hexagon NPU (the performance counters here cover the GPU's
engines only), so the sign that the NPU is doing the work is the model answering while the fork's processes
use almost no CPU: that is what the third graph shows.
"""
from __future__ import annotations

import base64
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

SCRIPT = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    "$ci = [System.Globalization.CultureInfo]::InvariantCulture; "
    "while ($true) { "
    "$c = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime; "
    "$m = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes; "
    "$p = @(Get-Process ollama*, llama-server*); "
    "$s = ($p | ForEach-Object { $_.TotalProcessorTime.TotalSeconds } | Measure-Object -Sum).Sum; "
    "[Console]::Out.WriteLine([string]::Format($ci, '{0} {1} {2} {3}', [double]$c, [double]$m, [double]$s, $p.Count)); "
    "[Console]::Out.Flush(); "
    "Start-Sleep -Milliseconds 1000 }"
)


@dataclass(frozen=True)
class Sample:
    cpu_percent: float  # the whole machine, 0..100
    available_mb: float
    fork_seconds: float  # processor time the fork's processes have used so far
    fork_processes: int


def parse_sample(line: str) -> Optional[Sample]:
    parts = line.split()
    if len(parts) != 4:
        return None
    try:
        return Sample(float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3])))
    except ValueError:
        return None


class Monitor:
    """Starts the sampler and hands each Sample to `on_sample` from a reader thread."""

    def __init__(self, on_sample: Callable[[Sample], None], popen: Callable[..., Any] = subprocess.Popen,
                 platform: str = sys.platform) -> None:
        self.on_sample = on_sample
        self.popen = popen
        self.platform = platform
        self.process: Any = None

    def start(self) -> bool:
        """False when sampling is not possible here (not Windows, or PowerShell cannot start)."""
        if self.platform != "win32":
            return False
        encoded = base64.b64encode(SCRIPT.encode("utf-16-le")).decode("ascii")
        try:
            self.process = self.popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                      text=True, encoding="utf-8", errors="replace",
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self.process = None
            return False
        threading.Thread(target=self._read, name="temperum-monitor", daemon=True).start()
        return True

    def _read(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self.process is None:  # stopped
                break
            sample = parse_sample(line)
            if sample is not None:
                self.on_sample(sample)

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
