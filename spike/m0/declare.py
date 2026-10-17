"""Write a schema-1 ollama_npu.json for a converted bundle, derived from its actual files.

Stages/providers come from genai_config.json, context from the config, processor name from
this machine's registry, runtime versions from the compile environment's target profile,
and the source claim from the fetched snapshot's _source.json.
"""

from __future__ import annotations

import argparse
import json
import winreg
from pathlib import Path


def processor_name() -> str:
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()


def stages(cfg: dict) -> list[dict]:
    decoder = cfg["model"]["decoder"]
    default = _provider(decoder.get("session_options", {}))
    out = []
    for group in decoder.get("pipeline", []):
        for _name, stage in group.items():
            provider = _provider(stage["session_options"]) if "session_options" in stage else default
            out.append({"filename": stage["filename"], "provider": provider})
    if not out:
        out.append({"filename": decoder["filename"], "provider": default})
    return out


def _provider(session: dict) -> str:
    for group in session.get("provider_options", []) or []:
        for name in group:
            return name.lower()
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--source", required=True, help="_source.json written by fetch_model.py")
    ap.add_argument("--precision", required=True, help="e.g. w8a16-rtn")
    ap.add_argument("--genai-version", default="0.12.2")
    ap.add_argument("--ort-version", default="1.24.1")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    cfg = json.loads((bundle / "genai_config.json").read_text(encoding="utf-8"))
    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    decl = {
        "schema_version": 1,
        "platform": "windows/arm64",
        "provider": "qnn",
        "genai_version": args.genai_version,
        "ort_version": args.ort_version,
        "processor_names": [processor_name()],
        "context_length": cfg["model"]["context_length"],
        "precision": args.precision,
        "capabilities": ["completion"],
        "stages": stages(cfg),
        "source": {"repository": f"https://huggingface.co/{src['repo']}", "revision": src["commit"]},
    }
    (bundle / "ollama_npu.json").write_text(json.dumps(decl, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(decl, indent=2))


if __name__ == "__main__":
    main()
