"""An error answer from the fork's server reaches the person with the server's own message."""
from __future__ import annotations

import http.server
import threading

import pytest

from temperum.core.errors import TemperumError
from temperum.install import ollama_import


def test_error_text_prefers_the_json_error_field_and_stays_one_line() -> None:
    assert ollama_import.error_text(b'{"error":"template: :1: function \\"foo\\" not defined\\n  at line 1"}') == \
        'template: :1: function "foo" not defined at line 1'
    assert ollama_import.error_text(b"<html>Bad Gateway</html>") == "<html>Bad Gateway</html>"
    assert ollama_import.error_text(b'{"error": 5}') == '{"error": 5}'
    assert len(ollama_import.error_text(b'{"error":"' + b"x" * 1000 + b'"}')) == 400


def test_api_error_names_what_the_server_said() -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = b'{"error":"runner process has terminated: exit status 2"}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(TemperumError) as exc:
            ollama_import._api(f"http://127.0.0.1:{server.server_port}", "/api/chat", 5, {"model": "m"})
        assert exc.value.reason_code == "LOCAL_SERVER_ERROR"
        assert "HTTP 500: runner process has terminated: exit status 2" in str(exc.value)
    finally:
        server.shutdown()
