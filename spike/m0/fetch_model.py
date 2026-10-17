"""Download a Hugging Face model snapshot pinned to an immutable commit.

Uses huggingface_hub's standard authentication (existing login or HF_TOKEN), so gated
repositories work once access has been granted. Only weights, configs, tokenizer and
license/card files are fetched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PATTERNS = [
    "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken",
    "LICENSE*", "NOTICE*", "README*", "USE_POLICY*",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    info = HfApi().model_info(args.repo, revision=args.revision)
    start = time.time()
    path = snapshot_download(
        args.repo, revision=info.sha, local_dir=args.out, allow_patterns=PATTERNS,
    )
    files = sorted(p.name for p in Path(path).iterdir() if p.is_file())
    meta = {
        "repo": args.repo, "requested_revision": args.revision, "commit": info.sha,
        "gated": getattr(info, "gated", None), "license": (info.card_data or {}).get("license")
        if info.card_data else None, "files": files, "seconds": round(time.time() - start, 1),
    }
    Path(args.out, "_source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
