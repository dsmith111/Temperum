"""Upload a model to ollama.com through the fork's server.

The server holds the account key (`~/.ollama/id_ed25519`) and signs the registry requests itself, so
temperum never sees a secret: it asks the server who the account is (`/api/me`), shows the sign-in link
when the server is not linked yet, names the model `account/name:tag`, and streams `/api/push`.
"""
from __future__ import annotations

import http.client
import json
import re
from typing import Any, Callable
from urllib.parse import urlsplit

from temperum.core.errors import TemperumError

SIGNIN_FALLBACK = "https://ollama.com/connect"
MAX_LINE = 1 << 16
_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
Progress = Callable[[str, int, int], None]


class SigninRequired(TemperumError):
    """The fork's server is not linked to an ollama.com account yet; `url` links it."""

    def __init__(self, url: str) -> None:
        super().__init__("push", "SIGNIN_REQUIRED", "The fork's server is not linked to an ollama.com account.",
                         hint=f"Open {url} to link it (this registers the server's key with your account), "
                              "then try again.")
        self.url = url


def _request(endpoint: str, path: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(str(url.hostname), url.port, timeout=timeout)
    try:
        connection.request("POST", path, body=json.dumps(body).encode("utf-8"),
                           headers={"Accept": "application/json", "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read(1 << 20)
    except (OSError, http.client.HTTPException) as exc:
        raise TemperumError("push", "LOCAL_SERVER_UNAVAILABLE", f"Cannot reach the fork at {endpoint}: {exc}") from exc
    finally:
        connection.close()


def account(endpoint: str, timeout: float = 60.0, request: Callable[..., tuple[int, bytes]] = _request) -> str:
    """The ollama.com account the server is linked to. Only the name is kept; the reply also carries the email."""
    status, data = request(endpoint, "/api/me", {}, timeout)
    try:
        value = json.loads(data or b"{}")
    except ValueError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    if status == 401:
        raise SigninRequired(str(value.get("signin_url") or SIGNIN_FALLBACK))
    if status != 200:
        raise TemperumError("push", "ACCOUNT_UNAVAILABLE",
                            f"The fork could not check its ollama.com account (HTTP {status}: "
                            f"{value.get('error', 'no detail')}).", hint="Is the machine online?")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise TemperumError("push", "ACCOUNT_UNAVAILABLE", "The fork reported an account without a name.")
    return name


def remote_name(account_name: str, local_tag: str) -> str:
    """`account/name:tag` for a local tag, keeping its tag and dropping any local namespace."""
    bare = local_tag.rsplit("/", 1)[-1]
    name, _, tag = bare.partition(":")
    return f"{account_name}/{name}:{tag or 'latest'}"


def check_remote_name(name: str) -> str:
    if len(name) > 160 or not re.fullmatch(rf"{_COMPONENT}/{_COMPONENT}(?::{_COMPONENT})?", name):
        raise TemperumError("push", "INVALID_NAME", f"{name!r} is not an ollama.com model name.",
                            hint="Use account/name or account/name:tag.")
    return name if ":" in name else name + ":latest"


def copy(endpoint: str, source: str, destination: str, timeout: float = 120.0,
         request: Callable[..., tuple[int, bytes]] = _request) -> None:
    """Give the local model its ollama.com name (the blobs are shared, nothing is duplicated)."""
    status, data = request(endpoint, "/api/copy", {"source": source, "destination": destination}, timeout)
    if status != 200:
        raise TemperumError("push", "COPY_FAILED", f"Could not name {source} as {destination} (HTTP {status}: "
                                                   f"{data[:200].decode('utf-8', 'replace')}).")


def push(endpoint: str, name: str, *, progress: Progress, timeout: float = 3600.0) -> dict[str, Any]:
    """Stream `/api/push`; `progress(status, completed, total)` follows each line the server sends."""
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(str(url.hostname), url.port, timeout=timeout)
    last = "starting"
    try:
        connection.request("POST", "/api/push", body=json.dumps({"model": name, "stream": True}).encode("utf-8"),
                           headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(4096).decode("utf-8", "replace")
            raise TemperumError("push", "PUSH_FAILED", f"/api/push returned HTTP {response.status}: {detail[:300]}")
        while True:
            line = response.readline(MAX_LINE)
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise TemperumError("push", "PUSH_FAILED", f"ollama.com refused the push: {event['error']}",
                                    hint="The name must start with your account, and the account must be linked "
                                         "to this server (ollama signin).")
            last = str(event.get("status") or last)
            progress(last, int(event.get("completed") or 0), int(event.get("total") or 0))
    except (OSError, http.client.HTTPException) as exc:
        raise TemperumError("push", "LOCAL_SERVER_UNAVAILABLE", f"Lost the fork at {endpoint}: {exc}") from exc
    finally:
        connection.close()
    if last != "success":
        raise TemperumError("push", "PUSH_FAILED", f"The push ended with status {last!r}.")
    return {"name": name, "status": last, "url": "https://ollama.com/" + name.rsplit(":", 1)[0]}
