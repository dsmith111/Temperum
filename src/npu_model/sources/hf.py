from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from npu_model.core.errors import NpuModelError


@dataclass(frozen=True)
class HfSpec:
    repo_id: str
    revision: Optional[str]


def parse_hf_spec(spec: str) -> HfSpec:
    # format: hf:org/repo[@rev]
    if not spec.startswith("hf:"):
        raise NpuModelError(stage="source", reason_code="HF_BAD_SPEC", message=f"Not an HF spec: {spec}")
    body = spec[len("hf:"):]
    if "@" in body:
        repo, rev = body.split("@", 1)
        return HfSpec(repo_id=repo, revision=rev)
    return HfSpec(repo_id=body, revision=None)


def materialize_hf(spec: HfSpec, cache_dir: Optional[Path]) -> Path:
    try:
        local_dir = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(local_dir).resolve()
    except Exception as e:
        raise NpuModelError(
            stage="source",
            reason_code="HF_DOWNLOAD_FAILED",
            message=f"Failed to download HF model: {spec.repo_id}@{spec.revision or 'main'}",
            hint="Check repo id / auth / network. Try `huggingface-cli login` if needed.",
            cause=e,
        ) from e
