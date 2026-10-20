from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from temperum.core.errors import TemperumError
from temperum.install import fork
from temperum.install import push as pushing
from temperum.install.validate import validate


# -- the fork's server ---------------------------------------------------------------------------
def test_endpoint_for_accepts_bare_hosts_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert fork.endpoint_for(None) == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:18767")
    assert fork.endpoint_for(None) == "http://127.0.0.1:18767"
    assert fork.endpoint_for("http://127.0.0.1:5") == "http://127.0.0.1:5"
    with pytest.raises(TemperumError):
        fork.endpoint_for("http://example.com:11434")


def test_ensure_uses_a_running_server(tmp_path: Path) -> None:
    info = fork.ensure("http://127.0.0.1:11434", None, lambda _m: None, log_dir=tmp_path,
                       probe=lambda *a, **k: "0.0.0-npu.dev.abc")
    assert info.version.endswith("abc") and not info.started and info.is_fork
    assert not fork.ServerInfo("http://127.0.0.1:11434", "0.12.3", False).is_fork


def test_ensure_starts_the_fork_when_none_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    answers = iter([None, None, "0.0.0-npu.dev.abc"])
    launched: dict[str, Any] = {}

    class Proc:
        pid = 4242
        returncode = None

        def poll(self) -> None:
            return None

    def popen(args: list[str], **kwargs: Any) -> Proc:
        launched.update(args=args, env=kwargs["env"], cwd=kwargs["cwd"])
        return Proc()

    said: list[str] = []
    info = fork.ensure("http://127.0.0.1:18799", exe, said.append, log_dir=tmp_path / "logs", popen=popen,
                       probe=lambda *a, **k: next(answers), sleep=lambda _s: None)
    assert info.started and info.pid == 4242 and launched["args"] == [str(exe), "serve"]
    assert launched["env"]["OLLAMA_HOST"] == "127.0.0.1:18799" and launched["env"]["OLLAMA_HEXAGON"] == "1"
    assert launched["cwd"] == str(tmp_path) and info.log == tmp_path / "logs" / "serve-18799.log"
    assert said and "starting the fork's server" in said[0]


def test_ensure_keeps_the_users_switch_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("OLLAMA_HEXAGON", "0")
    launched: dict[str, Any] = {}

    class Proc:
        pid = 1
        returncode = None

        def poll(self) -> None:
            return None

    def popen(args: list[str], **kwargs: Any) -> Proc:
        launched.update(env=kwargs["env"])
        return Proc()

    fork.ensure("http://127.0.0.1:18799", exe, lambda _m: None, log_dir=tmp_path, popen=popen,
                probe=lambda *a, **k: None if not launched else "v", sleep=lambda _s: None)
    assert launched["env"]["OLLAMA_HEXAGON"] == "0"


def test_ensure_without_an_executable_says_how_to_start_one(tmp_path: Path) -> None:
    with pytest.raises(TemperumError) as exc:
        fork.ensure("http://127.0.0.1:18799", None, lambda _m: None, log_dir=tmp_path, probe=lambda *a, **k: None)
    assert exc.value.reason_code == "SERVER_UNAVAILABLE" and "TEMPERUM_OLLAMA_EXE" in str(exc.value.hint)


def test_ensure_reports_a_server_that_exits(tmp_path: Path) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")

    class Proc:
        pid = 1
        returncode = 3

        def poll(self) -> int:
            return 3

    with pytest.raises(TemperumError) as exc:
        fork.ensure("http://127.0.0.1:18799", exe, lambda _m: None, log_dir=tmp_path,
                    popen=lambda *a, **k: Proc(), probe=lambda *a, **k: None, sleep=lambda _s: None)
    assert exc.value.reason_code == "SERVER_EXITED"


# -- validation through the fork -----------------------------------------------------------------
def test_validate_measures_through_the_fork() -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def api(endpoint: str, path: str, timeout: float, body: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append((path, body))
        if path == "/api/ps":
            return {"models": [{"name": "m:npu", "size": 1000, "size_vram": 1000, "accelerator": "NPU"}]}
        if path == "/api/chat":
            n = sum(1 for p, _ in calls if p == "/api/chat")
            content = {1: "Paris.", 2: "Berlin."}.get(n, "The sky is blue because of Rayleigh scattering. " * 10)
            return {"message": {"role": "assistant", "content": content}, "eval_count": 120,
                    "eval_duration": 6_000_000_000, "prompt_eval_count": 20, "prompt_eval_duration": 50_000_000}
        return {}

    cpu = iter([(10.0, 2), (10.6, 2)])
    report = validate("http://127.0.0.1:1", "m:npu", fork_dir=Path("C:/fork"), say=lambda _m: None, api=api,
                      cpu=lambda _d: next(cpu))
    assert report["placement"]["where"] == "npu" and report["placement"]["accelerator"] == "NPU"
    assert report["chat"] == {"answers": ["Paris.", "Berlin."], "correct": 2, "of": 2}
    assert report["speed"]["decode_tokens_per_s"] == 20.0 and report["speed"]["prompt_tokens_per_s"] == 400.0
    assert report["cpu"]["processes"] == 2 and report["cpu"]["cores"] > 0
    assert calls[-1] == ("/api/generate", {"model": "m:npu", "keep_alive": 0})


def test_validate_says_why_cpu_use_was_not_measured() -> None:
    def api(endpoint: str, path: str, timeout: float, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"message": {"content": "x"}, "models": []}

    report = validate("http://127.0.0.1:1", "m", fork_dir=None, say=lambda _m: None, api=api, cpu=lambda _d: None)
    assert report["placement"]["where"] == "unknown" and "TEMPERUM_OLLAMA_EXE" in report["cpu"]["skipped"]
    assert report["chat"]["correct"] == 0


# -- upload through the fork ---------------------------------------------------------------------
def test_account_name_and_signin() -> None:
    linked = json.dumps({"name": "dsmith", "email": "private@example.com"}).encode()
    assert pushing.account("http://127.0.0.1:1", request=lambda *a: (200, linked)) == "dsmith"
    with pytest.raises(pushing.SigninRequired) as exc:
        pushing.account("http://127.0.0.1:1", request=lambda *a: (
            401, b'{"error":"unauthorized","signin_url":"https://ollama.com/connect?name=pc&key=k"}'))
    assert exc.value.url.startswith("https://ollama.com/connect") and exc.value.reason_code == "SIGNIN_REQUIRED"
    with pytest.raises(TemperumError) as failed:
        pushing.account("http://127.0.0.1:1", request=lambda *a: (503, b'{"error":"account unavailable"}'))
    assert failed.value.reason_code == "ACCOUNT_UNAVAILABLE"


def test_remote_names() -> None:
    assert pushing.remote_name("dsmith", "llama-3.2-3b-instruct:npu") == "dsmith/llama-3.2-3b-instruct:npu"
    assert pushing.remote_name("dsmith", "local/thing") == "dsmith/thing:latest"
    assert pushing.check_remote_name("dsmith/thing") == "dsmith/thing:latest"
    for bad in ("thing", "a/b/c", "a/b:c:d", "registry.ollama.ai/a/b", "a/b c"):
        with pytest.raises(TemperumError):
            pushing.check_remote_name(bad)


def _serve(lines: list[bytes]) -> tuple[http.server.HTTPServer, list[tuple[str, Any]]]:
    seen: list[tuple[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            seen.append((self.path, json.loads(body)))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for line in lines:
                self.wfile.write(line + b"\n")
                self.wfile.flush()

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_push_streams_progress_and_ends_on_success() -> None:
    server, seen = _serve([b'{"status":"retrieving manifest"}',
                           b'{"status":"pushing abc","digest":"sha256:abc","total":100,"completed":40}',
                           b'{"status":"pushing abc","digest":"sha256:abc","total":100,"completed":100}',
                           b'{"status":"success"}'])
    events: list[tuple[str, int, int]] = []
    try:
        result = pushing.push(f"http://127.0.0.1:{server.server_port}", "dsmith/thing:npu",
                              progress=lambda s, c, t: events.append((s, c, t)))
    finally:
        server.shutdown()
    assert result == {"name": "dsmith/thing:npu", "status": "success", "url": "https://ollama.com/dsmith/thing"}
    assert events[1] == ("pushing abc", 40, 100)
    assert seen[0] == ("/api/push", {"model": "dsmith/thing:npu", "stream": True})


def test_push_reports_the_registrys_refusal() -> None:
    server, _seen = _serve([b'{"error":"you are not authorized to push to this namespace"}'])
    try:
        with pytest.raises(TemperumError) as exc:
            pushing.push(f"http://127.0.0.1:{server.server_port}", "x/y:z", progress=lambda *a: None)
    finally:
        server.shutdown()
    assert exc.value.reason_code == "PUSH_FAILED" and "not authorized" in str(exc.value)


def test_copy_names_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    def request(endpoint: str, path: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        sent.append((path, body))
        return 200, b""

    pushing.copy("http://127.0.0.1:1", "thing:npu", "dsmith/thing:npu", request=request)
    assert sent == [("/api/copy", {"source": "thing:npu", "destination": "dsmith/thing:npu"})]
    with pytest.raises(TemperumError):
        pushing.copy("http://127.0.0.1:1", "a", "b/c", request=lambda *a: (404, b"not found"))
