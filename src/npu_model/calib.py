"""Pinned text corpora for calibration and held-out quality checks, without `datasets`.

pyarrow (required by `datasets`) has no Windows ARM64 build, so rows come from the Hugging
Face dataset-viewer API as JSON and are cached as JSONL with a provenance record (dataset
revision, row range, content hash). Text is used locally only and never shipped in bundles.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from npu_model.core.errors import NpuModelError

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASETS_API = "https://huggingface.co/api/datasets"
PAGE = 100  # dataset-viewer maximum page size


@dataclass(frozen=True)
class CorpusSpec:
    dataset: str = "Salesforce/wikitext"
    config: str = "wikitext-2-raw-v1"
    split: str = "train"
    rows: int = 3000
    field: str = "text"
    license: str = "CC BY-SA 3.0"


CALIBRATION = CorpusSpec()
HELDOUT = CorpusSpec(split="test", rows=1500)


@dataclass(frozen=True)
class Corpus:
    path: Path
    spec: CorpusSpec
    revision: str
    sha256: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), **asdict(self.spec), "revision": self.revision,
                "sha256": self.sha256, "rows": self.rows}


def _get_json(url: str, timeout: float = 60) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "npu-model"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return json.loads(response.read().decode("utf-8"))


def ensure_corpus(spec: CorpusSpec, root: Path, get_json: Callable[[str], Any] = _get_json) -> Corpus:
    """Return the cached corpus for `spec`, fetching it once at the dataset's current revision."""
    try:
        revision = str(get_json(f"{DATASETS_API}/{spec.dataset}")["sha"])
    except Exception as exc:  # noqa: BLE001
        raise NpuModelError("calibration", "CORPUS_UNAVAILABLE", f"Could not reach the dataset API: {exc}",
                            hint="Calibration text is fetched once and cached; check the network.") from exc
    stem = f"{spec.dataset.replace('/', '--')}--{spec.config}--{spec.split}-{spec.rows}-{revision[:12]}"
    path = root / f"{stem}.jsonl"
    meta = root / f"{stem}.meta.json"
    if path.is_file() and meta.is_file():
        record = json.loads(meta.read_text(encoding="utf-8"))
        return Corpus(path, spec, record["revision"], record["sha256"], record["rows"])
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    lines: list[str] = []
    for offset in range(0, spec.rows, PAGE):
        query = urllib.parse.urlencode({"dataset": spec.dataset, "config": spec.config, "split": spec.split,
                                        "offset": offset, "length": min(PAGE, spec.rows - offset)})
        page = None
        for attempt in range(3):
            try:
                page = get_json(f"{ROWS_API}?{query}")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise NpuModelError("calibration", "CORPUS_UNAVAILABLE",
                                        f"Fetching calibration rows failed: {exc}") from exc
                time.sleep(2 * (attempt + 1))
        assert page is not None
        for row in page.get("rows", []):
            line = json.dumps({"text": row["row"].get(spec.field, "")}, ensure_ascii=False)
            lines.append(line)
            digest.update(line.encode("utf-8") + b"\n")
    if not lines:
        raise NpuModelError("calibration", "CORPUS_EMPTY", f"No rows returned for {spec.dataset}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    corpus = Corpus(path, spec, revision, digest.hexdigest(), len(lines))
    meta.write_text(json.dumps(corpus.to_dict(), indent=2), encoding="utf-8")
    return corpus
