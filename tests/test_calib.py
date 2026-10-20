from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from temperum.calib import CorpusSpec, ensure_corpus
from temperum.core.errors import TemperumError

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


class FakeApi:
    def __init__(self, fail_rows: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_rows = fail_rows

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if "/api/datasets/" in url:
            return {"sha": REVISION}
        if self.fail_rows:
            raise OSError("handshake failure")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        offset, length = int(query["offset"][0]), int(query["length"][0])
        return {"rows": [{"row": {"text": f"line {i}"}} for i in range(offset, offset + length)]}


def test_corpus_is_fetched_in_pages_and_recorded(tmp_path: Path) -> None:
    api = FakeApi()
    corpus = ensure_corpus(CorpusSpec(rows=250), tmp_path, get_json=api)
    lines = corpus.path.read_text(encoding="utf-8").splitlines()
    assert corpus.rows == 250 and len(lines) == 250
    assert json.loads(lines[0]) == {"text": "line 0"} and json.loads(lines[-1]) == {"text": "line 249"}
    assert corpus.revision == REVISION and len(corpus.sha256) == 64
    assert sum("rows?" in c for c in api.calls) == 3  # 100 + 100 + 50
    meta = json.loads(corpus.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == corpus.sha256 and meta["license"] == "CC BY-SA 3.0"


def test_corpus_is_reused_for_the_same_revision(tmp_path: Path) -> None:
    ensure_corpus(CorpusSpec(rows=100), tmp_path, get_json=FakeApi())
    api = FakeApi()
    again = ensure_corpus(CorpusSpec(rows=100), tmp_path, get_json=api)
    assert again.rows == 100 and not any("rows?" in c for c in api.calls)


def test_unreachable_dataset_api_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("temperum.calib.time.sleep", lambda _s: None)
    with pytest.raises(TemperumError) as exc:
        ensure_corpus(CorpusSpec(rows=100), tmp_path, get_json=FakeApi(fail_rows=True))
    assert exc.value.reason_code == "CORPUS_UNAVAILABLE"
    assert not list(tmp_path.glob("*.jsonl"))
