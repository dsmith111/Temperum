"""The fork's server: find the one conversions install into, or start one for this session.

`install`, `validate` and `push` all go through a running server. When none answers at the endpoint
and the fork executable is known, temperum starts `ollama serve` detached, with the NPU switch on unless
the environment already says otherwise, and leaves it running for `ollama run` afterwards.
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from temperum import settings
from temperum.core.errors import TemperumError
from temperum.host import HEXAGON_OPT_IN
from temperum.install.ollama_import import local_endpoint

DEFAULT_HOST = "http://127.0.0.1:11434"
EXE_SETTING = "TEMPERUM_OLLAMA_EXE"
Popen = Callable[..., Any]


@dataclass(frozen=True)
class ServerInfo:
    endpoint: str
    version: str
    started: bool  # True when this process launched it
    pid: int | None = None
    log: Path | None = None

    @property
    def is_fork(self) -> bool:
        return "npu" in self.version.lower()


def executable(ollama_exe: Path | None) -> Path | None:
    """The fork executable from the option, the environment or the stored setting, when it names a file."""
    exe, _source = settings.ollama_exe(ollama_exe)
    return exe if exe is not None and exe.is_file() else None


def endpoint_for(host: str | None) -> str:
    """The local endpoint: --host, else OLLAMA_HOST (with or without a scheme), else the stored one, else
    Ollama's default."""
    chosen = str(settings.resolve("host", host, DEFAULT_HOST).value)
    if "://" not in chosen:
        chosen = "http://" + chosen
    return local_endpoint(chosen)


def version(endpoint: str, timeout: float = 3.0) -> str | None:
    """The server's version string when one answers at the endpoint."""
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(str(url.hostname), url.port, timeout=timeout)
    try:
        connection.request("GET", "/api/version", headers={"Accept": "application/json"})
        response = connection.getresponse()
        data = response.read(4096)
        if response.status != 200:
            return None
        value = json.loads(data)
        found = value.get("version") if isinstance(value, dict) else None
        return str(found) if found else None
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        connection.close()


def ensure(endpoint: str, exe: Path | None, say: Callable[[str], None], *, log_dir: Path, wait_s: float = 90,
           popen: Popen = subprocess.Popen, probe: Callable[..., str | None] = version,
           sleep: Callable[[float], None] = time.sleep, switch_on: bool = False) -> ServerInfo:
    """A server answering at the endpoint: the running one, or one started from the fork executable.

    A started server gets the NPU switch unless the environment already decides it; `switch_on` forces it.
    """
    found = probe(endpoint)
    if found:
        return ServerInfo(endpoint, found, started=False)
    if exe is None:
        raise TemperumError("import", "SERVER_UNAVAILABLE", f"No Ollama server answers at {endpoint}.",
                            hint=f"Start the fork's server ({HEXAGON_OPT_IN}=1, then `ollama serve`), or run "
                                 "`temperum setup --ollama-exe PATH` once so temperum can start it.")
    url = urlsplit(endpoint)
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{url.hostname}:{url.port}"
    if switch_on:
        env[HEXAGON_OPT_IN] = "1"
    else:
        env.setdefault(HEXAGON_OPT_IN, "1")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"serve-{url.port}.log"
    say(f"starting the fork's server at {endpoint} with {HEXAGON_OPT_IN}={env[HEXAGON_OPT_IN]} (log: {log})")
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if sys.platform == "win32" else 0
    try:
        with log.open("ab") as handle:
            process = popen([str(exe), "serve"], env=env, stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT, cwd=str(exe.parent), creationflags=flags,
                            start_new_session=sys.platform != "win32")
    except OSError as exc:
        raise TemperumError("import", "SERVER_LAUNCH_FAILED", f"Cannot start {exe}: {exc}") from exc
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        found = probe(endpoint, timeout=2.0)
        if found:
            return ServerInfo(endpoint, found, started=True, pid=int(process.pid), log=log)
        if process.poll() is not None:
            raise TemperumError("import", "SERVER_EXITED",
                                f"The fork's server exited with code {process.returncode}; see {log}.",
                                hint="Another server may hold the port; pass --host with a free one.")
        sleep(0.5)
    raise TemperumError("import", "SERVER_TIMEOUT", f"The fork's server did not answer within {wait_s:.0f} s; "
                                                    f"see {log}.")


# -- a server started without the switch ---------------------------------------------------------
Run = Callable[..., Any]


def listener(endpoint: str, run: Run = subprocess.run) -> tuple[int, Path] | None:
    """The process listening on the endpoint's port: its id and executable (Windows only)."""
    if sys.platform != "win32":
        return None
    port = urlsplit(endpoint).port
    script = (f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
              "Select-Object -First 1; if ($c) { $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue; "
              "if ($p -and $p.Path) { Write-Output ($p.Id.ToString() + '|' + $p.Path) } }")
    try:
        result = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                     text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    pid, sep, path = str(result.stdout).strip().partition("|")
    if not sep or not pid.isdigit() or not path:
        return None
    return int(pid), Path(path)


def same_folder(path: Path, exe: Path) -> bool:
    """Whether a process executable lives in the known fork's folder (the server and its runners do)."""
    try:
        return str(path.resolve().parent).lower() == str(exe.resolve().parent).lower()
    except OSError:
        return False


def stop_tree(folder: Path, run: Run = subprocess.run) -> int:
    """Stop every process running from the fork's folder (its server and runners); how many were stopped."""
    if sys.platform != "win32":
        return 0
    prefix = str(folder).rstrip("\\") + "\\"
    script = ("$p = @(Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith('"
              + prefix.replace("'", "''") + "') }); $p | Stop-Process -Force -ErrorAction SilentlyContinue; "
              "Write-Output $p.Count")
    try:
        result = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                     text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0
    text = str(result.stdout).strip()
    return int(text) if text.isdigit() else 0


def restart_with_switch(endpoint: str, exe: Path, say: Callable[[str], None], *, log_dir: Path,
                        run: Run = subprocess.run, popen: Popen = subprocess.Popen,
                        probe: Callable[..., str | None] = version,
                        sleep: Callable[[float], None] = time.sleep, wait_s: float = 20) -> ServerInfo:
    """Stop the fork's server at the endpoint, when it runs from the known executable's folder, and start one
    with the NPU switch on."""
    found = listener(endpoint, run)
    if found is None or not same_folder(found[1], exe):
        raise TemperumError("import", "SERVER_NOT_OURS",
                            f"The server at {endpoint} is not the fork at {exe}, so temperum leaves it alone.",
                            hint="Stop it yourself, then run `temperum validate TAG`: temperum starts the fork's "
                                 "server with the NPU switch on.")
    say(f"stopping the fork's server at {endpoint} (pid {found[0]}) and its runners")
    stop_tree(exe.parent, run)
    deadline = time.monotonic() + wait_s
    while probe(endpoint, timeout=2.0) is not None and time.monotonic() < deadline:
        sleep(0.5)
    if probe(endpoint, timeout=2.0) is not None:
        raise TemperumError("import", "SERVER_STILL_RUNNING", f"Something still answers at {endpoint}.",
                            hint="Stop it yourself, then run `temperum validate TAG`.")
    return ensure(endpoint, exe, say, log_dir=log_dir, popen=popen, probe=probe, sleep=sleep, switch_on=True)
