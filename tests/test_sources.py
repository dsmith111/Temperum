from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.sources import ModelRef, ResolvedModel, fetch, parse_ref, resolve

COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
CONFIG = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_hidden_layers": 24}


class FakeApi:
    def __init__(self, error: Exception | None = None, files: tuple[str, ...] = ("config.json", "model.safetensors")):
        self.error = error
        self.files = files
        self.calls: list[tuple[Any, ...]] = []

    def model_info(self, repo: str, revision: str | None = None, files_metadata: bool = False) -> Any:
        self.calls.append((repo, revision, files_metadata))
        if self.error:
            raise self.error
        siblings = [SimpleNamespace(rfilename=f, size=1000 if f.endswith(".safetensors") else 10) for f in self.files]
        return SimpleNamespace(sha=COMMIT, gated=False, card_data={"license": "apache-2.0"}, siblings=siblings)


def _download_to(tmp_path: Path):  # type: ignore[no-untyped-def]
    def download(repo: str, filename: str, revision: str | None = None) -> str:
        assert revision == COMMIT
        path = tmp_path / filename
        path.write_text(json.dumps(CONFIG), encoding="utf-8")
        return str(path)

    return download


@pytest.mark.parametrize("text,repo,revision", [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", None),
    ("hf:Qwen/Qwen2.5-0.5B-Instruct@main", "Qwen/Qwen2.5-0.5B-Instruct", "main"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct@" + COMMIT, "HuggingFaceTB/SmolLM2-360M-Instruct", COMMIT),
])
def test_parse_hf_refs(text: str, repo: str, revision: str | None) -> None:
    ref = parse_ref(text)
    assert (ref.kind, ref.repo, ref.revision) == ("hf", repo, revision)


def test_parse_local_folder_requires_config(tmp_path: Path) -> None:
    with pytest.raises(NpuModelError) as exc:
        parse_ref(str(tmp_path))
    assert exc.value.reason_code == "LOCAL_MISSING_CONFIG"
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    ref = parse_ref(str(tmp_path))
    assert ref.kind == "local" and ref.path == tmp_path.resolve()


@pytest.mark.parametrize("text", ["not a model", "https://huggingface.co/Qwen/x", "justname"])
def test_parse_rejects_non_refs(text: str) -> None:
    with pytest.raises(NpuModelError) as exc:
        parse_ref(text)
    assert exc.value.reason_code == "BAD_MODEL_REF"


def test_resolve_pins_commit_and_reads_config_without_weights(tmp_path: Path) -> None:
    api = FakeApi()
    model = resolve(parse_ref("Qwen/Qwen2.5-0.5B-Instruct"), api=api, download=_download_to(tmp_path))
    assert model.commit == COMMIT and model.license == "apache-2.0" and not model.gated
    assert (model.architecture, model.model_type, model.weight_bytes) == ("Qwen2ForCausalLM", "qwen2", 1000)
    assert model.name == "qwen2.5-0.5b-instruct"
    assert api.calls == [("Qwen/Qwen2.5-0.5B-Instruct", None, True)]


def test_resolve_rejects_repos_without_safetensors(tmp_path: Path) -> None:
    api = FakeApi(files=("config.json", "pytorch_model.bin"))
    with pytest.raises(NpuModelError) as exc:
        resolve(parse_ref("org/model"), api=api, download=_download_to(tmp_path))
    assert exc.value.reason_code == "NO_SAFETENSORS"


def test_gated_repo_error_explains_how_to_get_access(tmp_path: Path) -> None:
    import httpx
    from huggingface_hub import errors

    url = "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json"
    gated = errors.GatedRepoError("403 gated", response=httpx.Response(403, request=httpx.Request("GET", url)))
    with pytest.raises(NpuModelError) as exc:
        resolve(parse_ref("meta-llama/Llama-3.2-1B-Instruct"), api=FakeApi(error=gated),
                download=_download_to(tmp_path))
    assert exc.value.reason_code == "HF_GATED"
    assert "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct" in (exc.value.hint or "")


def test_fetch_records_source_and_reuses_snapshot(tmp_path: Path) -> None:
    model = ResolvedModel(ModelRef("hf", repo="Qwen/Qwen2.5-0.5B-Instruct"), commit=COMMIT, gated=False,
                          license="apache-2.0", config=CONFIG, weight_bytes=1000)
    calls: list[dict[str, Any]] = []

    def snapshot(repo: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        Path(kwargs["local_dir"], "model.safetensors").write_bytes(b"x")
        return str(kwargs["local_dir"])

    dest = fetch(model, tmp_path, snapshot=snapshot)
    assert dest == tmp_path / "Qwen--Qwen2.5-0.5B-Instruct" / COMMIT
    record = json.loads((dest / "_source.json").read_text(encoding="utf-8"))
    assert record["commit"] == COMMIT and record["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert calls[0]["revision"] == COMMIT and "*.safetensors" in calls[0]["allow_patterns"]
    assert fetch(model, tmp_path, snapshot=snapshot) == dest and len(calls) == 1


def test_fetch_drops_only_the_repeated_xet_notice(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    model = ResolvedModel(ModelRef("hf", repo="Qwen/Qwen2.5-0.5B-Instruct"), commit=COMMIT, gated=False,
                          license="apache-2.0", config=CONFIG, weight_bytes=1000)
    log = logging.getLogger("huggingface_hub.file_download")

    def snapshot(repo: str, **kwargs: Any) -> str:
        for _ in range(3):
            log.warning("Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back")
        log.warning("Some other download problem")
        return str(kwargs["local_dir"])

    with caplog.at_level(logging.WARNING, logger="huggingface_hub.file_download"):
        fetch(model, tmp_path, snapshot=snapshot)
    assert [r.getMessage() for r in caplog.records] == ["Some other download problem"]
    assert not log.filters  # the filter is removed afterwards


def test_local_model_resolves_without_network(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    model = resolve(parse_ref(str(tmp_path)))
    assert model.commit is None and model.weight_bytes == 3 and model.model_type == "qwen2"
    assert fetch(model, tmp_path / "unused") == tmp_path.resolve()
