"""Run `olive run` and record wall time, peak process-tree memory and per-pass timing.

Writes <log>.summary.json next to the full log so M0 measurements are reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

PASS_RE = re.compile(r"Running pass (\S+)")


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "olive", "run", "--config", args.config],
        cwd=args.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    ps = psutil.Process(proc.pid)
    peak = {"rss": 0}
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            try:
                peak["rss"] = max(peak["rss"], _tree_rss(ps))
            except psutil.NoSuchProcess:
                break
            time.sleep(1.0)

    threading.Thread(target=sample, daemon=True).start()
    passes: list[dict] = []
    with log_path.open("w", encoding="utf-8") as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            m = PASS_RE.search(line)
            if m:
                now = time.time() - start
                if passes:
                    passes[-1]["seconds"] = round(now - passes[-1]["start"], 1)
                passes.append({"pass": m.group(1), "start": round(now, 1)})
                print(f"[{now:8.1f}s] {m.group(1)}", flush=True)
    code = proc.wait()
    done.set()
    total = time.time() - start
    if passes and "seconds" not in passes[-1]:
        passes[-1]["seconds"] = round(total - passes[-1]["start"], 1)
    summary = {
        "exit_code": code, "seconds": round(total, 1),
        "peak_rss_gb": round(peak["rss"] / 2**30, 2), "passes": passes,
    }
    Path(str(log_path) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
