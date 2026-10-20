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
    """The fork executable from the option or the setting, when it names a file."""
    setting = os.environ.get(EXE_SETTING)
    exe = ollama_exe or (Path(setting) if setting else None)
    return exe if exe is not None and exe.is_file() else None


def endpoint_for(host: str | None) -> str:
    """The local endpoint to use: --host, else OLLAMA_HOST (with or without a scheme), else Ollama's default."""
    chosen = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
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
           sleep: Callable[[float], None] = time.sleep) -> ServerInfo:
    """A server answering at the endpoint: the running one, or one started from the fork executable."""
    found = probe(endpoint)
    if found:
        return ServerInfo(endpoint, found, started=False)
    if exe is None:
        raise TemperumError("import", "SERVER_UNAVAILABLE", f"No Ollama server answers at {endpoint}.",
                            hint=f"Start the fork's server ({HEXAGON_OPT_IN}=1, then `ollama serve`), or set "
                                 f"{EXE_SETTING} to the fork's ollama.exe so temperum can start it.")
    url = urlsplit(endpoint)
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{url.hostname}:{url.port}"
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
