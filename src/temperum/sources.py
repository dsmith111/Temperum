"""Resolve and fetch source models (Hugging Face or a local folder) pinned to exact content.

Uses huggingface_hub's standard authentication (an existing login or HF_TOKEN), so gated
repositories work once the account has been granted access. Tokens are never recorded.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from temperum.core.errors import TemperumError

# Weights, configs, tokenizers and licence/card files; never pickles or remote code.
SNAPSHOT_PATTERNS = [
    "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken",
    "LICENSE*", "NOTICE*", "README*", "USE_POLICY*",
]
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# Config keys only speculative-decoding draft checkpoints carry (DFlash, EAGLE, Medusa).
DRAFT_KEYS = ("dflash_config", "eagle_config", "draft_config", "target_layer_ids", "num_target_layers",
              "draft_vocab_size", "target_hidden_size", "medusa_num_heads")


@dataclass(frozen=True)
class ModelRef:
    kind: str  # "hf" | "local"
    repo: str | None = None
    revision: str | None = None
    path: Path | None = None

    @property
    def display(self) -> str:
        if self.kind == "local":
            return str(self.path)
        return f"{self.repo}@{self.revision}" if self.revision else str(self.repo)


@dataclass
class ResolvedModel:
    ref: ModelRef
    commit: str | None
    gated: bool
    license: str | None
    config: dict[str, Any]
    weight_bytes: int | None
    files: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Lowercase short name, e.g. qwen2.5-0.5b-instruct."""
        base = self.ref.repo.split("/")[-1] if self.ref.repo else Path(str(self.ref.path)).name
        return re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-.") or "model"

    @property
    def architecture(self) -> str | None:
        archs = self.config.get("architectures") or []
        return str(archs[0]) if archs else None

    @property
    def model_type(self) -> str | None:
        value = self.config.get("model_type")
        return str(value) if value else None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.ref.kind, "repo": self.ref.repo, "revision": self.ref.revision,
                "path": str(self.ref.path) if self.ref.path else None, "commit": self.commit,
                "gated": self.gated, "license": self.license, "architecture": self.architecture,
                "model_type": self.model_type, "weight_bytes": self.weight_bytes}


def draft_model_reason(config: dict[str, Any], tags: list[str] | None = None) -> str | None:
    """Why a checkpoint is a speculative-decoding draft (a helper for a larger target model), or None."""
    archs = [str(a) for a in (config.get("architectures") or [])]
    if any("draft" in a.lower() for a in archs):
        return f"architecture {archs[0]}"
    keys = [k for k in DRAFT_KEYS if k in config]
    if keys:
        return "config has " + ", ".join(keys)
    if "draft-model" in (tags or []):
        return "tagged draft-model on Hugging Face"
    return None


def base_model_tag(tags: list[str] | None) -> str | None:
    """The repo a Hugging Face `base_model:org/name` tag names, if any."""
    for tag in tags or []:
        if tag.startswith("base_model:") and tag.count(":") == 1:
            return tag.split(":", 1)[1]
    return None


def _refuse_draft(ref: ModelRef, config: dict[str, Any], tags: list[str] | None) -> None:
    reason = draft_model_reason(config, tags)
    if reason is None:
        return
    base = base_model_tag(tags)
    raise TemperumError("source", "DRAFT_MODEL",
                        f"{ref.display} is a speculative-decoding draft model ({reason}): a helper that proposes "
                        "tokens for a larger target model and cannot answer on its own.",
                        hint=(f"Convert the target model instead: {base}" if base else
                              "Convert the target model it was trained for instead")
                        + " (check its size against this machine first: `temperum convert MODEL --plan`).")


def parse_ref(text: str) -> ModelRef:
    """`org/name[@rev]`, `hf:org/name[@rev]`, or a local folder containing config.json."""
    raw = text.strip()
    if not raw.startswith("hf:"):
        local = Path(raw).expanduser()
        if local.is_dir():
            if not (local / "config.json").is_file():
                raise TemperumError("source", "LOCAL_MISSING_CONFIG", f"No config.json in {local}",
                                    hint="Point at a Hugging Face-style model folder (config.json + weights).")
            return ModelRef("local", path=local.resolve())
    body = raw[3:] if raw.startswith("hf:") else raw
    repo, _, revision = body.partition("@")
    if not _REPO_RE.match(repo):
        raise TemperumError("source", "BAD_MODEL_REF", f"Not a Hugging Face repo id or local folder: {text}",
                            hint="Use org/name (optionally @revision) or a path to a local model folder.")
    return ModelRef("hf", repo=repo, revision=revision or None)


def _hf_error(exc: Exception, ref: ModelRef) -> TemperumError:
    from huggingface_hub import errors

    url = f"https://huggingface.co/{ref.repo}"
    if isinstance(exc, errors.GatedRepoError):
        return TemperumError("source", "HF_GATED", f"{ref.repo} is gated and this account has not been granted access.",
                             hint=f"Request access at {url}, then sign in (`hf auth login`) or set HF_TOKEN.",
                             cause=exc)
    if isinstance(exc, errors.RevisionNotFoundError):
        return TemperumError("source", "HF_REVISION_NOT_FOUND", f"Revision {ref.revision!r} not found in {ref.repo}.",
                             cause=exc)
    if isinstance(exc, errors.RepositoryNotFoundError):
        return TemperumError("source", "HF_NOT_FOUND", f"Model repository not found: {ref.repo}",
                             hint="Check the id; private repositories also need `hf auth login`.", cause=exc)
    return TemperumError("source", "HF_UNAVAILABLE", f"Could not reach Hugging Face for {ref.repo}: {exc}",
                         hint="Check the network or proxy settings and retry.", cause=exc)


def resolve(ref: ModelRef, api: Any = None, download: Callable[..., str] | None = None) -> ResolvedModel:
    """Metadata only: immutable commit, licence, gating and config.json (no weights downloaded)."""
    if ref.kind == "local":
        assert ref.path is not None
        config = json.loads((ref.path / "config.json").read_text(encoding="utf-8"))
        files = sorted(p.name for p in ref.path.iterdir() if p.is_file())
        weights = sum(p.stat().st_size for p in ref.path.glob("*.safetensors"))
        _refuse_draft(ref, config, None)
        return ResolvedModel(ref, commit=None, gated=False, license=None, config=config,
                             weight_bytes=weights or None, files=files)
    from huggingface_hub import HfApi, hf_hub_download

    api = api or HfApi()
    download = download or hf_hub_download
    try:
        info = api.model_info(ref.repo, revision=ref.revision, files_metadata=True)
        config_path = download(ref.repo, "config.json", revision=info.sha)
    except Exception as exc:  # noqa: BLE001 - mapped to actionable errors
        raise _hf_error(exc, ref) from exc
    siblings = list(getattr(info, "siblings", None) or [])
    files = sorted(s.rfilename for s in siblings)
    weights = sum(int(getattr(s, "size", 0) or 0) for s in siblings if s.rfilename.endswith(".safetensors"))
    card = getattr(info, "card_data", None)
    license_id = card.get("license") if card is not None and hasattr(card, "get") else None
    if not any(name.endswith(".safetensors") for name in files):
        raise TemperumError("source", "NO_SAFETENSORS", f"{ref.repo} has no .safetensors weights.",
                            hint="Only safetensors checkpoints are converted (pickled weights are never loaded).")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _refuse_draft(ref, config, [str(t) for t in (getattr(info, "tags", None) or [])])
    return ResolvedModel(ref, commit=info.sha, gated=bool(getattr(info, "gated", False)),
                         license=str(license_id) if license_id else None, config=config,
                         weight_bytes=weights or None, files=files)


def fetch(model: ResolvedModel, root: Path, snapshot: Callable[..., str] | None = None) -> Path:
    """Download (or reuse) the pinned snapshot; returns a folder with a _source.json record."""
    if model.ref.kind == "local":
        assert model.ref.path is not None
        return model.ref.path
    from huggingface_hub import snapshot_download

    snapshot = snapshot or snapshot_download
    assert model.ref.repo is not None and model.commit is not None
    dest = root / model.ref.repo.replace("/", "--") / model.commit
    record = dest / "_source.json"
    if record.is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    downloads = logging.getLogger("huggingface_hub.file_download")
    downloads.addFilter(_no_xet_notice)
    try:
        snapshot(model.ref.repo, revision=model.commit, local_dir=str(dest), allow_patterns=SNAPSHOT_PATTERNS)
    except Exception as exc:  # noqa: BLE001
        raise _hf_error(exc, model.ref) from exc
    finally:
        downloads.removeFilter(_no_xet_notice)
    record.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return dest


def _no_xet_notice(record: logging.LogRecord) -> bool:
    # Without the optional hf_xet package, huggingface_hub repeats this for every file of a Xet-backed
    # repo; it then downloads over plain HTTP, which works.
    return not record.getMessage().startswith("Xet Storage is enabled for this repo, but the 'hf_xet' package")
