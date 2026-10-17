"""Fetch a pinned calibration corpus without the `datasets` package.

pyarrow (required by `datasets`) has no Windows ARM64 build, so rows come from the
Hugging Face dataset-viewer API as plain JSON and are written to JSONL together with
a small provenance record (dataset, config, split, revision, row range, content hash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
HUB = "https://huggingface.co/api/datasets"
PAGE = 100  # API maximum per request


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "npu-model-spike"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    revision = _get(f"{HUB}/{args.dataset}")["sha"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    kept = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for offset in range(0, args.rows, PAGE):
            query = urllib.parse.urlencode({
                "dataset": args.dataset, "config": args.config, "split": args.split,
                "offset": offset, "length": min(PAGE, args.rows - offset),
            })
            for attempt in range(3):
                try:
                    page = _get(f"{API}?{query}")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for row in page["rows"]:
                line = json.dumps({"text": row["row"]["text"]}, ensure_ascii=False)
                fh.write(line + "\n")
                digest.update(line.encode("utf-8") + b"\n")
                kept += 1
    meta = {
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "revision": revision, "rows": kept, "sha256": digest.hexdigest(),
        "source": API, "license": "CC BY-SA 3.0 (calibration use only; not redistributed)",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
