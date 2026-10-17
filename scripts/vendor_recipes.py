"""Vendor the maintained olive-recipes QNN NPU recipes at a pinned commit (maintainer tool).

Writes src/npu_model/recipes/upstream/<org>--<name>.json and upstream/index.json (source
path + sha256 per recipe). Preference per model: QNN/x_elite_config.json, QNN/config.json,
then the AI Toolkit (aitk/) QNN config. GPU-only and docker variants are skipped.

Usage: python scripts/vendor_recipes.py --commit <full sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "microsoft/olive-recipes"
DEST = Path(__file__).resolve().parents[1] / "src" / "npu_model" / "recipes" / "upstream"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "npu-model-vendor"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return bytes(response.read())


def _rank(path: str) -> int | None:
    lowered = path.lower()
    if "gpu" in lowered or "docker" in lowered or not lowered.endswith(".json"):
        return None
    if lowered.endswith("/qnn/x_elite_config.json"):
        return 0
    if lowered.endswith("/qnn/config.json"):
        return 1
    if "/aitk/" in lowered and "qnn" in lowered.rsplit("/", 1)[-1]:
        return 2
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{args.commit}?recursive=1"))["tree"]
    best: dict[str, tuple[int, str]] = {}
    for item in tree:
        rank = _rank(item["path"]) if item["type"] == "blob" else None
        if rank is None:
            continue
        model_dir = item["path"].split("/", 1)[0]
        if model_dir not in best or rank < best[model_dir][0]:
            best[model_dir] = (rank, item["path"])
    DEST.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, str]] = {}
    for _model_dir, (_rank_value, path) in sorted(best.items()):
        raw = _get(f"https://raw.githubusercontent.com/{REPO}/{args.commit}/{path}")
        config = json.loads(raw)
        input_model = config.get("input_model", {})
        if input_model.get("type") != "HfModel" or not any(
                p.get("type") == "EPContextBinaryGenerator" for p in config.get("passes", {}).values()):
            continue  # not an LLM NPU recipe (vision/encoder models, etc.)
        repo_id = str(input_model["model_path"])
        name = repo_id.replace("/", "--") + ".json"
        (DEST / name).write_bytes(raw)
        index[repo_id] = {"file": name, "upstream_path": path, "sha256": hashlib.sha256(raw).hexdigest()}
        print(f"{repo_id:45s} <- {path}")
    meta = {"repository": f"https://github.com/{REPO}", "commit": args.commit, "license": "MIT",
            "recipes": index}
    (DEST / "index.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
