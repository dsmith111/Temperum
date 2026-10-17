from pathlib import Path

import pytest

from npu_model.sources import hf


def test_hf_download_uses_current_supported_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(*, repo_id: str, revision: str | None, cache_dir: str | None) -> str:
        assert repo_id == "example/model"
        assert revision == "revision"
        assert cache_dir == str(tmp_path)
        return str(tmp_path)

    monkeypatch.setattr(hf, "snapshot_download", download)
    assert hf.materialize_hf(hf.HfSpec("example/model", "revision"), tmp_path) == tmp_path.resolve()
