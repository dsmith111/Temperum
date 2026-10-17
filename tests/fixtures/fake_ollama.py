"""Subprocess control-flow fixture only, never a runtime/native certification mock."""
import hashlib
import json
import os
import sys
from pathlib import Path

capture = Path(sys.argv[1])
args = sys.argv[2:]
if len(args) != 4 or args[0] != "create" or args[2] != "-f":
    sys.exit(9)
data = Path(args[3]).read_bytes().decode("utf-8")
source = data.splitlines()[0].removeprefix('FROM "').removesuffix('"')
bundle = Path(source)
hashes = {path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in bundle.rglob("*") if path.is_file()}
capture.write_text(json.dumps({"argv": args, "host": os.environ.get("OLLAMA_HOST"), "modelfile": data,
                               "auth": os.environ.get("OLLAMA_AUTH"), "no_proxy": os.environ.get("NO_PROXY"),
                               "asset_hashes": hashes}))
print("Fake create fixture only; native execution not tested.", file=sys.stderr)
