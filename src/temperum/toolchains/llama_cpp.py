"""llama.cpp's Hugging Face to GGUF converter, pinned to the commit of the runtime's llama.cpp.

The converter is plain Python (convert_hf_to_gguf.py, its `conversion` package and the `gguf`
package). It is fetched once from the source archive of the pinned commit, and every file is
checked against a vendored manifest (llama_cpp_pins/<tag>.json, made from the git objects), so a
changed archive cannot change what runs. Files under llama_cpp_patches/<tag>/ then replace their
upstream counterparts: small fixes the pinned converter needs (each says why), recorded in every
bundle's manifest. It runs in the quantize environment, which already
has torch, transformers and numpy; `gguf` comes from the same commit through PYTHONPATH, so
nothing is installed into that environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable

from temperum import paths
from temperum.core.errors import TemperumError

MANIFESTS = Path(__file__).resolve().parent / "llama_cpp_pins"
PATCHES = Path(__file__).resolve().parent / "llama_cpp_patches"
Opener = Callable[[str], IO[bytes]]


def _open_url(url: str) -> IO[bytes]:
    response: IO[bytes] = urllib.request.urlopen(url, timeout=60)  # noqa: S310 - pinned https URL
    return response


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConverterSource:
    tag: str
    commit: str
    archive: str
    files: dict[str, str]  # path inside the repository -> sha256
    patches: dict[str, Path] = field(default_factory=dict)  # path inside the repository -> temperum's replacement

    @classmethod
    def load(cls, tag: str) -> ConverterSource:
        path = MANIFESTS / f"{tag}.json"
        if not path.is_file():
            raise TemperumError("toolchain", "UNKNOWN_LLAMA_CPP", f"No pinned llama.cpp converter for {tag}")
        data = json.loads(path.read_text(encoding="utf-8"))
        root = PATCHES / tag
        patches = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()} \
            if root.is_dir() else {}
        return cls(data["tag"], data["commit"], data["archive"], dict(data["files"]), patches)

    @property
    def directory(self) -> Path:
        return paths.home() / "tools" / f"llama.cpp-{self.tag}"

    @property
    def script(self) -> Path:
        return self.directory / "convert_hf_to_gguf.py"

    def expected(self, name: str, upstream: bool = False) -> str:
        """The digest a file should have on disk: temperum's patch where one exists, else the pinned one."""
        if not upstream and name in self.patches:
            return _sha256(self.patches[name])
        return self.files[name]

    def problems(self, root: Path | None = None, upstream: bool = False) -> list[str]:
        """Files that are missing or differ from the manifest (with temperum's patches applied, unless
        `upstream`: a freshly extracted archive before patching)."""
        root = root or self.directory
        return [name for name in self.files
                if not (root / name).is_file() or _sha256(root / name) != self.expected(name, upstream)]

    def ensure(self, opener: Opener = _open_url, progress: Callable[[str], None] | None = None) -> Path:
        if not self.problems():
            return self.directory
        if progress:
            progress(f"fetching llama.cpp's GGUF converter ({self.tag}, {self.commit[:9]})")
        staging = self.directory.with_name(self.directory.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            with opener(self.archive) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    name = member.name.split("/", 1)[1] if "/" in member.name else ""
                    if name not in self.files or not member.isfile():
                        continue
                    data = archive.extractfile(member)
                    if data is None:
                        continue
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data.read())
        except (OSError, tarfile.TarError) as exc:
            raise TemperumError("toolchain", "LLAMA_CPP_FETCH_FAILED",
                                f"Could not fetch llama.cpp {self.tag} from {self.archive}: {exc}",
                                hint="Check the network connection to github.com and retry.") from exc
        bad = self.problems(staging, upstream=True)
        if bad:
            raise TemperumError("toolchain", "LLAMA_CPP_MISMATCH",
                                f"llama.cpp {self.tag} archive does not match the pinned files: "
                                f"{', '.join(bad[:5])}{' ...' if len(bad) > 5 else ''}",
                                hint=f"Partial download left at {staging}")
        for name, patch in self.patches.items():
            shutil.copyfile(patch, staging / name)
        if progress and self.patches:
            progress("applied temperum's converter patches: " + ", ".join(sorted(self.patches)))
        if self.directory.exists():
            shutil.rmtree(self.directory)
        staging.rename(self.directory)
        return self.directory

    def environment(self) -> dict[str, str]:
        """Environment for running the converter: offline, pinned gguf package, no token."""
        env = {**os.environ, "PYTHONPATH": str(self.directory / "gguf-py"), "PYTHONUTF8": "1",
               "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env.pop("HF_TOKEN", None)
        return env
