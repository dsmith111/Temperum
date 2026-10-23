"""Calibrated quantization: the importance-matrix stage, and everything that carries or shows it."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from temperum import flows, gguf
from temperum.calib import CALIBRATION, Corpus
from temperum.cli.main import app
from temperum.core.errors import TemperumError
from temperum.host import HostInfo
from temperum.optimize import evaluate, report
from temperum.optimize.space import Candidate
from temperum.profiles import get_target
from temperum.sources import ModelRef, ResolvedModel

X_ELITE = HostInfo(system="Windows", native_arch="arm64", python_platform="win-amd64",
                   processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU",
                   total_ram_gb=15.4, available_ram_gb=6.0)


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))


def _model() -> ResolvedModel:
    return ResolvedModel(ModelRef("hf", repo="Qwen/Qwen2.5-1.5B-Instruct"), commit="989aa7980e4c", gated=False,
                         license="apache-2.0", weight_bytes=3_087_467_144,
                         config={"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
                                 "torch_dtype": "bfloat16", "max_position_embeddings": 32768})


def _payload(root: Path, imatrix: bool = True) -> Path:
    """A fork package's lib/ollama, with or without the imatrix tool (packages before the target lack it)."""
    lib = root / "lib" / "ollama"
    (lib / "hexagon").mkdir(parents=True)
    exe = root / "ollama.exe"
    exe.write_bytes(b"")
    names = ["llama-quantize.exe", "llama-server.exe", "llama-quantize", "llama-server"]
    if imatrix:
        names += ["llama-imatrix.exe", "llama-imatrix"]
    for name in names:
        (lib / name).write_bytes(b"")
    (lib / "hexagon" / "ggml-hexagon.dll").write_bytes(b"")
    for name in gguf.HTP_LIBRARIES:
        (lib / "hexagon" / name).write_bytes(b"")
    return exe


def _version(stdout: str) -> Any:
    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout, "")
    return run


class _Proc:
    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


def test_quantize_passes_the_matrix_before_the_recipe_overrides(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def popen(cmd: list[str], **_kwargs: Any) -> _Proc:
        seen["cmd"] = cmd
        (tmp_path / "out.gguf.partial").write_bytes(b"gguf")
        return _Proc(["llama_model_quantize_impl: quant size = 888.43 MiB\n"])

    payload = gguf.LlamaPayload(tmp_path / "lib")
    gguf.quantize(payload, tmp_path / "src.gguf", tmp_path / "out.gguf", "Q4_0", tmp_path / "log", lambda _m: None,
                  popen=popen, overrides=["--tensor-type", "attn_v=q8_0"], imatrix=tmp_path / "imatrix.gguf")
    assert seen["cmd"][1:6] == ["--allow-requantize", "--imatrix", str(tmp_path / "imatrix.gguf"),
                                "--tensor-type", "attn_v=q8_0"]
    assert seen["cmd"][-1] == "Q4_0" and (tmp_path / "out.gguf").is_file()


def test_compute_imatrix_runs_the_runtimes_tool_on_the_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GGML_BACKEND_PATH", str(tmp_path / "ggml-hexagon.dll"))  # a verify run may leave it set
    seen: dict[str, Any] = {}
    said: list[str] = []

    def popen(cmd: list[str], env: dict[str, str] | None = None, **_kwargs: Any) -> _Proc:
        seen["cmd"], seen["env"] = cmd, env
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"GGUF")
        return _Proc(["compute_imatrix: computing over 200 chunks, n_ctx=512, batch_size=2048, n_seq=4\n",
                      "compute_imatrix: 9.47 seconds per pass - ETA 7.88 minutes\n"])

    payload = gguf.LlamaPayload(tmp_path / "lib")
    out = gguf.compute_imatrix(payload, tmp_path / "model-Q8_0.gguf", tmp_path / "calibration.txt",
                               tmp_path / "imatrix.gguf", tmp_path / "log", said.append, popen=popen)
    assert out == tmp_path / "imatrix.gguf" and out.is_file()
    cmd = seen["cmd"]
    assert cmd[0] == str(payload.imatrix) and cmd[cmd.index("-m") + 1] == str(tmp_path / "model-Q8_0.gguf")
    assert cmd[cmd.index("-f") + 1] == str(tmp_path / "calibration.txt")
    assert cmd[cmd.index("-o") + 1].endswith(".partial.gguf")  # the tool writes GGUF only to a .gguf name
    assert cmd[cmd.index("--chunks") + 1] == str(gguf.CALIBRATION_CHUNKS) and "--no-ppl" in cmd
    assert cmd[-2:] == ["-ngl", "0"] and "GGML_BACKEND_PATH" not in seen["env"]
    assert said == ["compute_imatrix: 9.47 seconds per pass - ETA 7.88 minutes"]


def test_a_failed_calibration_says_so_with_the_log(tmp_path: Path) -> None:
    def popen(_cmd: list[str], **_kwargs: Any) -> _Proc:
        return _Proc(["error: could not load the model\n"], returncode=1)

    with pytest.raises(TemperumError) as caught:
        gguf.compute_imatrix(gguf.LlamaPayload(tmp_path / "lib"), tmp_path / "m.gguf", tmp_path / "c.txt",
                             tmp_path / "imatrix.gguf", tmp_path / "log", lambda _m: None, popen=popen)
    assert caught.value.reason_code == "GGUF_CALIBRATE_FAILED" and "exit 1" in caught.value.message
    assert caught.value.hint and "could not load the model" in caught.value.hint


def test_calibration_text_turns_the_cached_rows_into_plain_text(tmp_path: Path) -> None:
    corpus = tmp_path / "rows.jsonl"
    corpus.write_text('{"text": " = Title = \\n"}\n{"text": ""}\n\n{"text": "A body line."}\n', encoding="utf-8")
    out = gguf.calibration_text(corpus, tmp_path / "calibration.txt")
    assert out.read_text(encoding="utf-8") == " = Title = \nA body line.\n"
    assert gguf.calibration_text(corpus, out) == out  # kept once written


def test_the_calibration_record_names_the_tool_and_the_corpus_but_not_a_local_path(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "rows.jsonl", CALIBRATION, "deadbeef", "ab" * 32, 3000)
    record = gguf.calibration_record(gguf.LlamaPayload(tmp_path / "lib"), corpus)
    assert record["tool"] == str(tmp_path / "lib" / gguf._exe("llama-imatrix"))
    assert record["chunks"] == gguf.CALIBRATION_CHUNKS and record["context"] == gguf.CALIBRATION_CONTEXT
    assert record["corpus"]["dataset"] == "Salesforce/wikitext" and record["corpus"]["split"] == "train"
    assert "path" not in record["corpus"] and record["corpus"]["revision"] == "deadbeef"


def test_the_plan_calibrates_when_the_runtime_ships_the_tool(tmp_path: Path) -> None:
    run = _version("version: 0.6.0-dev (build 1, commit e467e5b)")
    target = get_target("gguf-hexagon")
    exe = _payload(tmp_path / "new")
    calibrated = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "a")
    plain = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "b", calibrate=False)
    assert calibrated.calibrate and calibrated.to_dict()["calibrate"] is True
    assert not plain.calibrate and plain.to_dict()["calibrate"] is False
    assert calibrated.estimates["minutes"] > plain.estimates["minutes"]
    assert calibrated.job_id != plain.job_id  # a calibrated file is a different file
    assert not any("llama-imatrix" in w for w in calibrated.warnings)
    q8 = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "c", quant="Q8_0")
    assert not q8.calibrate and not any("llama-imatrix" in w for w in q8.warnings)  # nothing to calibrate

    old = _payload(tmp_path / "old", imatrix=False)
    legacy = gguf.plan(_model(), X_ELITE, target, ollama_exe=old, run=run, out=tmp_path / "d")
    assert not legacy.calibrate
    assert any("no llama-imatrix" in w and "not calibrated" in w for w in legacy.warnings)
    assert legacy.job_id == plain.job_id  # uncalibrated jobs keep their earlier ids


def test_the_search_quantizes_every_candidate_with_the_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def quantize(_payload: Any, _src: Path, out: Path, _quant: str, _log: Path, _say: Any,
                 overrides: Any = (), imatrix: Path | None = None, **_kwargs: Any) -> Path:
        seen["imatrix"], seen["overrides"] = imatrix, list(overrides)
        out.write_bytes(b"gguf")
        return out

    monkeypatch.setattr(evaluate.gguf, "quantize", quantize)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"text": "some held-out text"}\n', encoding="utf-8")
    ev = evaluate.Evaluator(gguf.LlamaPayload(tmp_path / "lib"), tmp_path / "src.gguf", tmp_path / "work", heldout,
                            context=4096, chunks=16, say=lambda _m: None, imatrix=tmp_path / "imatrix.gguf")
    ev._build(Candidate.upgrade(["attn_v"]))
    assert seen == {"imatrix": tmp_path / "imatrix.gguf", "overrides": ["--tensor-type", "attn_v=q8_0"]}
    assert ev._build(Candidate.plain("Q8_0")) == tmp_path / "src.gguf"  # the anchor is the source itself


def test_the_recipe_page_shows_how_to_reproduce_a_calibrated_mix() -> None:
    chosen = {"label": "Q4_0, attn_v at Q8_0", "why": "better", "quantize_args": ["--tensor-type", "attn_v=q8_0"],
              "base": "Q4_0", "name": "q4_0+attn_v.q8_0"}
    doc = {"chosen": chosen, "trials": [], "source": {"calibration": {
        "chunks": 200, "corpus": {"dataset": "Salesforce/wikitext", "split": "train"}}}}
    page = report.markdown(doc)
    assert "`--allow-requantize --imatrix imatrix.gguf --tensor-type attn_v=q8_0 <Q8_0 source> <out> Q4_0`" in page
    assert "200 chunks of Salesforce/wikitext train rows" in page
    assert "--imatrix" not in report.markdown({**doc, "source": {}})


def test_the_plan_table_and_the_ready_lines_say_calibrated(tmp_path: Path) -> None:
    plan = SimpleNamespace(
        estimates={"params_billions": 1.5, "minutes": 9, "ram_gb": 3.1, "disk_gb": 4.2, "gguf_gb": 0.9},
        model=SimpleNamespace(ref=SimpleNamespace(display="org/model"), commit="989aa7980e4c", architecture="X",
                              license="mit", gated=False, chat_template=None),
        target=SimpleNamespace(id="gguf-hexagon", description="the NPU"), runtime=SimpleNamespace(description="fork"),
        payload=SimpleNamespace(root=tmp_path, htp=tmp_path), quant="Q4_0", context=4096, recipe=None,
        verify_level="standard", destination=tmp_path / "out", calibrate=True)
    console = Console(file=io.StringIO(), width=220)
    console.print(flows.gguf_plan_table(plan))
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Calibration" in text and f"{gguf.CALIBRATION_CHUNKS} chunks" in text and "wikitext-2 train" in text
    console = Console(file=io.StringIO(), width=220)
    console.print(flows.gguf_plan_table(SimpleNamespace(**{**vars(plan), "calibrate": False})))
    assert "Calibration" not in console.file.getvalue()  # type: ignore[attr-defined]

    summary = flows.gguf_summary(SimpleNamespace(bundle=tmp_path, job_dir=tmp_path, verification={"verdict": "passed"},
                                                 recipe=None, calibration={"chunks": 200}), start=0.0)
    assert summary["calibrated"] is True
    assert any("calibrated with an importance matrix" in line for line in flows.ready_lines(summary))
    plain = flows.gguf_summary(SimpleNamespace(bundle=tmp_path, job_dir=tmp_path, verification={"verdict": "passed"},
                                               recipe=None, calibration=None), start=0.0)
    assert plain["calibrated"] is False and not any("importance matrix" in line for line in flows.ready_lines(plain))


def test_convert_takes_no_calibrate_and_calibrates_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("temperum.cli.main._convert_gguf", lambda model, target, **kwargs: seen.append(kwargs))
    runner = CliRunner()
    assert runner.invoke(app, ["convert", "org/model", "--plan", "--target", "gguf-hexagon", "--no-calibrate"]).exit_code == 0
    assert runner.invoke(app, ["convert", "org/model", "--plan", "--target", "gguf-hexagon"]).exit_code == 0
    assert [k["calibrate"] for k in seen] == [False, True]
