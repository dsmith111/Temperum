"""Turn Olive's output folder into an installable NPU bundle.

Next to the GenAI pipeline the bundle gets:
  genai_config.json        context and max_length set to the declared context; a pad token
                           that never occurs in prompts (GenAI treats in-prompt pad ids as padding)
  ollama_npu.json          the fork's compatibility declaration (schema 1)
  Modelfile                FROM . plus the generated Go template and stop sequences
  LICENSE / NOTICE / ...   the source model's licence files and card, unchanged
  npu_model_manifest.json  provenance and per-file hashes
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from npu_model import __version__
from npu_model.bundle.inspect import Inspection, inspect_prebuilt
from npu_model.core.errors import NpuModelError
from npu_model.olive_run import pipeline_files
from npu_model.profiles import RuntimeProfile, TargetProfile
from npu_model.templates import ChatTemplate, load_jinja

LICENSE_PREFIXES = ("license", "notice", "use_policy", "readme")
# Olive copies the source config/tokenizer files; weights or caches never belong in a bundle.
SKIP_SUFFIXES = {".safetensors", ".bin.index.json", ".pt", ".pth", ".ckpt"}
MANIFEST = "npu_model_manifest.json"
MIN_CONTEXT = 128


@dataclass(frozen=True)
class AssembleInputs:
    olive_output: Path
    snapshot: Path  # source model folder (licence files, tokenizer_config.json)
    destination: Path
    target: TargetProfile
    runtime: RuntimeProfile
    chat: ChatTemplate
    context: int | None  # None: min(model maximum, DEFAULT_CONTEXT)
    precision: str
    processor_name: str | None  # exact name for the declaration (None: no declaration)
    source: dict[str, Any]  # repo/commit/licence of the source model
    provenance: dict[str, Any]  # recipe, toolchains, calibration, quantization settings
    overwrite: bool = False


DEFAULT_CONTEXT = 4096  # long-prompt retrieval was verified to ~3.5K tokens in M0


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise NpuModelError("assemble", "BAD_JSON", f"{path.name} is not a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()  # never write through a hard link into Olive's cache
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _token_content(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) and value else None


def prompt_token_ids(tokenizer_config: dict[str, Any], jinja: str | None) -> set[int]:
    """Special tokens a chat prompt can contain: those the chat template writes."""
    if not jinja:
        return set()
    ids: set[int] = set()
    by_content: dict[str, int] = {}
    for key, spec in (tokenizer_config.get("added_tokens_decoder") or {}).items():
        content = _token_content(spec)
        if content is not None and str(key).isdigit():
            by_content[content] = int(key)
            if content in jinja:
                ids.add(int(key))
    for name in ("bos_token", "eos_token"):
        content = _token_content(tokenizer_config.get(name))
        if content and name in jinja and content in by_content:
            ids.add(by_content[content])
    return ids


def safe_pad_token(current: int | None, tokenizer_config: dict[str, Any], jinja: str | None) -> int | None:
    """A pad id that never appears in prompts (SmolLM2 ships pad = <|im_end|>, which breaks decoding)."""
    used = prompt_token_ids(tokenizer_config, jinja)
    if current is not None and current not in used:
        return current
    candidates: list[tuple[int, int]] = []
    for key, spec in (tokenizer_config.get("added_tokens_decoder") or {}).items():
        content = (_token_content(spec) or "").lower()
        if not str(key).isdigit() or int(key) in used or not content:
            continue
        rank = 0 if "pad" in content else 1 if "endoftext" in content else 2 if "unk" in content else 3
        candidates.append((rank, int(key)))
    if not candidates:
        raise NpuModelError("assemble", "PAD_TOKEN_CONFLICT",
                            f"pad_token_id {current} occurs in chat prompts and no unused special token exists.")
    return min(candidates)[1]


def _stages(cfg: dict[str, Any]) -> list[dict[str, str]]:
    decoder = cfg["model"]["decoder"]

    def provider(session: dict[str, Any]) -> str:
        for group in session.get("provider_options", []) or []:
            for name in group:
                return str(name).lower()
        return "cpu"

    default = provider(decoder.get("session_options", {}))
    stages = []
    for group in decoder.get("pipeline", []):
        for spec in group.values():
            stages.append({"filename": spec["filename"],
                           "provider": provider(spec["session_options"]) if "session_options" in spec else default})
    return stages or [{"filename": decoder["filename"], "provider": default}]


def declaration(cfg: dict[str, Any], inputs: AssembleInputs) -> dict[str, Any]:
    decl: dict[str, Any] = {
        "schema_version": 1, "platform": "windows/arm64", "provider": inputs.target.provider,
        "genai_version": inputs.runtime.genai_version, "ort_version": inputs.runtime.ort_version,
        "processor_names": [inputs.processor_name], "context_length": cfg["model"]["context_length"],
        "precision": inputs.precision, "capabilities": ["completion"], "stages": _stages(cfg),
    }
    repo, commit = inputs.source.get("repo"), inputs.source.get("commit")
    if repo and commit:
        decl["source"] = {"repository": f"https://huggingface.co/{repo}", "revision": commit}
    return decl


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # same volume: no copy (Olive's cache holds the other link)
    except OSError:
        shutil.copy2(src, dst)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(inputs: AssembleInputs) -> tuple[Path, Inspection]:
    out = inputs.olive_output
    if not pipeline_files(out):
        raise NpuModelError("assemble", "OLIVE_OUTPUT_MISSING", f"No GenAI pipeline in {out}")
    final = inputs.destination
    if final.exists() and not inputs.overwrite:
        raise NpuModelError("assemble", "DESTINATION_EXISTS", f"{final} already exists.",
                            hint="Pass --overwrite to replace it, or --out to choose another folder.")
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for item in sorted(out.iterdir()):
        name = item.name.lower()
        if item.is_file() and not any(name.endswith(s) for s in SKIP_SUFFIXES) and name != "ollama_npu.json":
            _place(item, staging / item.name)

    cfg = _read_json(staging / "genai_config.json")
    model = cfg["model"]
    maximum = int(model.get("context_length") or DEFAULT_CONTEXT)
    window = int((model.get("decoder", {}).get("sliding_window") or {}).get("window_size") or 1)
    context = inputs.context or min(maximum, DEFAULT_CONTEXT)
    context -= context % window  # whole prefill windows only
    if not MIN_CONTEXT <= context <= maximum:
        raise NpuModelError("assemble", "BAD_CONTEXT", f"Context {context} is outside {MIN_CONTEXT}..{maximum}.")
    model["context_length"] = context
    cfg.setdefault("search", {})["max_length"] = context
    tokenizer_config = _read_json(inputs.snapshot / "tokenizer_config.json") if (
        inputs.snapshot / "tokenizer_config.json").is_file() else {}
    pad = safe_pad_token(model.get("pad_token_id"), tokenizer_config, load_jinja(inputs.snapshot))
    pad_changed = pad != model.get("pad_token_id")
    model["pad_token_id"] = pad
    _write_json(staging / "genai_config.json", cfg)

    if inputs.processor_name:
        _write_json(staging / "ollama_npu.json", declaration(cfg, inputs))
    (staging / "Modelfile").write_text(inputs.chat.modelfile(), encoding="utf-8", newline="\n")
    licences = []
    for item in sorted(inputs.snapshot.iterdir()):
        if item.is_file() and item.name.lower().startswith(LICENSE_PREFIXES) and not (staging / item.name).exists():
            shutil.copy2(item, staging / item.name)
            licences.append(item.name)

    files = {p.name: {"size": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(staging.iterdir()) if p.is_file()}
    manifest = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter": {"name": "npu-model", "version": __version__},
        "source": inputs.source,
        "target": {"id": inputs.target.id, "soc_model": inputs.target.soc_model, "htp_arch": inputs.target.htp_arch},
        "runtime": {"id": inputs.runtime.id, "genai": inputs.runtime.genai_version, "ort": inputs.runtime.ort_version,
                    "qnn": inputs.runtime.qnn_version},
        "precision": inputs.precision,
        "context_length": context,
        "chat_template": {"family": inputs.chat.family, "verified": inputs.chat.verified,
                          "default_system": inputs.chat.default_system, "stops": list(inputs.chat.stops)},
        "genai_config_changes": {"context_length": context, "max_length": context,
                                 **({"pad_token_id": pad} if pad_changed else {})},
        "licence_files": licences,
        "provenance": inputs.provenance,
        "files": files,
    }
    _write_json(staging / MANIFEST, manifest)

    inspection = inspect_prebuilt(staging, processor_name=inputs.processor_name)
    if not inspection.importable:
        codes = ", ".join(f"{e['code']}: {e['message']}" for e in inspection.errors) or inspection.target["status"]
        raise NpuModelError("assemble", "BUNDLE_INVALID", f"The assembled bundle failed inspection ({codes}).",
                            hint=f"Partial bundle left at {staging}")
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    inspection.input = str(final)
    return final, inspection
