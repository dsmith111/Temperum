from __future__ import annotations

from pathlib import Path

from temperum.core.manifest import sha256_file, collect_files, write_manifest


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    h = sha256_file(f)
    assert isinstance(h, str)
    assert len(h) == 64


def test_collect_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b", encoding="utf-8")

    files = collect_files(tmp_path)
    paths = [f["path"] for f in files]
    assert "a.txt" in paths
    assert "sub/b.txt" in paths


def test_write_manifest(tmp_path: Path) -> None:
    mp = tmp_path / "manifest.json"
    write_manifest(mp, {"tool": "test"})
    assert mp.exists()
    import json
    data = json.loads(mp.read_text(encoding="utf-8"))
    assert data["tool"] == "test"
