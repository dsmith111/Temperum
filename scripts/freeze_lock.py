"""Write a toolchain lock file from a working environment (maintainer tool).

Usage:
  python scripts/freeze_lock.py --env <venv> --role quantize --runtime fork-qnn-1.24.1 \
      --device cpu --torch-index https://download.pytorch.org/whl/cpu \
      --exclude onnxruntime-qnn --tested "X1E78100, M0 conversions" --out <lock.txt>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, type=Path)
    ap.add_argument("--role", required=True, choices=["quantize", "compile", "verify"])
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--torch-index", default="")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--tested", required=True, help="where and how this exact set was exercised")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    python = args.env / ("Scripts/python.exe" if (args.env / "Scripts").is_dir() else "bin/python")
    probe = subprocess.run([str(python), "-c", "import sys,sysconfig;"
                            "print(f'{sys.version_info[0]}.{sys.version_info[1]}', sysconfig.get_platform())"],
                           check=True, capture_output=True, text=True).stdout.split()
    freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all", "--disable-pip-version-check"],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    skip = {"pip", *(name.lower() for name in args.exclude)}
    pins = sorted((line for line in freeze if "==" in line and line.split("==")[0].lower() not in skip),
                  key=str.lower)
    header = [
        f"# npu-model toolchain lock: role={args.role} runtime={args.runtime} platform={probe[1]}"
        + (f" device={args.device}" if args.device else ""),
        "# mode: exact",
        f"# python: {probe[0]}",
        *([f"# torch-index: {args.torch_index}"] if args.torch_index else []),
        f"# tested: {args.tested}",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + pins) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.out}: {len(pins)} pins")


if __name__ == "__main__":
    main()
