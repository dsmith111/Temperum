"""Bounded, unauthenticated HF acquisition; independent files, not a Hub cache."""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, NoReturn
from urllib.parse import quote, urljoin, urlsplit

from npu_model.core.errors import NpuModelError
from npu_model.inspect.prebuilt import (
    CONSUMER, MAX_DEPTH, MAX_FILES, MAX_GRAPH, MAX_JSON, MAX_TOTAL,
    inspect_prebuilt, inventory, read_object, regular_path, relative_name,
)

ENDPOINT = "https://huggingface.co"
MAX_FILE = 64 << 30
MAX_SOURCE = 4 << 20
MAX_STATE = 16 << 20
CHUNK = 1 << 20
STAGES = {"downloading", "interrupted", "failed", "validating", "promoting", "completed"}
CDN_HOSTS = {
    "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
    "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
    "us.aws.cdn.hf.co", "eu.aws.cdn.hf.co",
}


def _fail(code: str, message: str) -> NoReturn:
    raise NpuModelError("acquire", code, message)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail("TIMEOUT", "Acquisition time budget expired; no completion is claimed.")
    return remaining


def _deadline(timeout: float) -> float:
    if not math.isfinite(timeout) or not 1 <= timeout <= 3600:
        _fail("INVALID_TIMEOUT", "Timeout must be finite and between 1 and 3600 seconds.")
    return time.monotonic() + timeout


@contextmanager
def _response(url: str, deadline: float, *, redirects: bool = False) -> Iterator[http.client.HTTPResponse]:
    # Unlike Hub SDK sessions, http.client has no token, proxy, netrc, cache,
    # telemetry, custom endpoint, automatic retry or redirect inheritance.
    for hop in range(6):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in CDN_HOSTS
                or parsed.netloc != parsed.hostname or parsed.fragment
                or len(url) > 16384 or any(ord(c) < 32 for c in url)):
            _fail("UNSAFE_REDIRECT", "Only credential-free HTTPS on the bounded HF/CDN host allowlist is supported.")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=min(30.0, _remaining(deadline)))
        try:
            connection.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""),
                               headers={"Accept-Encoding": "identity", "User-Agent": "npu-model-prebuilt/1"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not redirects or not location or hop == 5:
                    _fail("UNSAFE_REDIRECT", "Metadata redirects/repository substitution or excessive redirects refused.")
                url = urljoin(url, location)
                continue
            if response.status in {401, 403, 404}:
                _fail("HF_PUBLIC_ONLY", "Public unauthenticated repositories only; gated/private/missing inputs "
                      "are unsupported. Do not supply a token; obtain an authorized local folder separately.")
            if response.status != 200 or response.getheader("Content-Encoding", "identity") != "identity":
                _fail("HTTP_FAILED", f"HF returned HTTP {response.status} or unsupported content encoding.")
            yield response
            return
        except (OSError, http.client.HTTPException) as exc:
            raise NpuModelError("acquire", "TRANSFER_FAILED", "HF HTTPS request/transfer failed; retry explicitly.",
                                cause=exc) from exc
        finally:
            connection.close()
    _fail("UNSAFE_REDIRECT", "Redirect limit exceeded.")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("INVALID_METADATA", "Duplicate key in HF metadata.")
        result[key] = value
    return result


def _constant(value: str) -> NoReturn:
    _fail("INVALID_METADATA", f"Nonfinite number in HF metadata: {value}")


def _metadata(repo: str, revision: str, deadline: float) -> dict[str, Any]:
    url = f"{ENDPOINT}/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    with _response(url, deadline) as response:
        chunks: list[bytes] = []
        size = 0
        while size <= MAX_JSON:
            _remaining(deadline)
            chunk = response.read1(min(CHUNK, MAX_JSON + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        data = b"".join(chunks)
    if not data or len(data) > MAX_JSON:
        _fail("METADATA_LIMIT", "HF metadata must be nonempty and at most 4 MiB.")
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise NpuModelError("acquire", "INVALID_METADATA", "Invalid HF metadata JSON.", cause=exc) from exc
    if not isinstance(value, dict) or value.get("id") != repo:
        _fail("INVALID_METADATA", "HF response must identify exactly the requested repository.")
    if value.get("private") is not False or value.get("gated") is not False:
        _fail("HF_PUBLIC_ONLY", "Only explicitly public, ungated repositories are supported; no login/auth retry.")
    if not _hex(value.get("sha"), 40):
        _fail("INVALID_METADATA", "HF must resolve the revision to a full lowercase Git commit.")
    if _hex(revision, 40) and value["sha"] != revision:
        _fail("METADATA_DRIFT", "HF response substituted the requested immutable commit.")
    return value


def _safe_input(repo: str, revision: str, subfolder: str | None) -> None:
    component = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    if (not re.fullmatch(rf"{component}/{component}", repo) or ".." in repo or "--" in repo
            or any(part.endswith((".", ".git")) for part in repo.split("/"))):
        _fail("INVALID_REPO", "Use a literal public HF namespace/repository, not a URL, token or endpoint.")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", revision)
            or ".." in revision or any(part in {"", "."} for part in revision.split("/"))):
        _fail("INVALID_REVISION", "Use a bounded literal branch/tag/full commit, without traversal or URL syntax.")
    if subfolder is not None:
        relative_name(subfolder)
        if (len(subfolder.split("/")) > MAX_DEPTH
                or {".git", ".cache"} & {part.casefold() for part in subfolder.split("/")}):
            _fail("UNSAFE_SUBFOLDER", "Subfolder is outside the bounded portable folder contract.")


def _absent_destination(path: Path) -> Path:
    if ".." in path.parts:
        _fail("INVALID_DESTINATION", "Destination must not contain traversal components.")
    path = Path(os.path.abspath(path.expanduser()))
    relative_name(path.name)
    regular_path(path.parent, directory=True)
    if os.path.lexists(path):
        _fail("DESTINATION_EXISTS", "Destination already exists; it will never be removed or overwritten.")
    return path


@dataclass(frozen=True)
class RemoteFile:
    remote_path: str
    path: str
    size: int
    algorithm: str
    oid: str
    blob_id: str


def _attribution(name: str) -> bool:
    return re.fullmatch(r"(README|LICENSE|LICENCE|NOTICE|AUTHORS|COPYING|ATTRIBUTION|COPYRIGHT)"
                        r"(?:[._-][A-Za-z0-9._-]+)?", name, re.IGNORECASE) is not None


def _aliases(names: list[str]) -> None:
    seen: dict[str, str] = {}
    files: set[str] = set()
    directories: set[str] = set()
    for name in names:
        try:
            relative_name(name)
        except UnicodeError as exc:
            raise NpuModelError("acquire", "INVALID_METADATA", "Remote paths must contain valid Unicode.",
                                cause=exc) from exc
        parts = name.split("/")
        if len(parts) > MAX_DEPTH:
            _fail("INVENTORY_LIMIT", "Remote path depth exceeds the local inspection bound.")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            alias = unicodedata.normalize("NFC", prefix).casefold()
            if alias in seen and seen[alias] != prefix:
                _fail("PATH_ALIAS", f"Case/Unicode path alias: {name}")
            seen[alias] = prefix
            if index < len(parts):
                directories.add(prefix)
        if name in files:
            _fail("PATH_ALIAS", f"Duplicate remote path: {name}")
        files.add(name)
    if files & directories:
        _fail("PATH_ALIAS", "A remote path is both a file and a directory.")
    if len(files | directories) > MAX_FILES:
        _fail("INVENTORY_LIMIT", "Remote inventory exceeds the 4096-entry folder bound.")


def _files(metadata: dict[str, Any], subfolder: str | None) -> tuple[RemoteFile, ...]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or not 0 < len(siblings) <= MAX_FILES:
        _fail("INVENTORY_LIMIT", "Expected 1..4096 HF repository file entries.")
    if any(not isinstance(item, dict) or not isinstance(item.get("rfilename"), str) for item in siblings):
        _fail("INVALID_METADATA", "HF file entries require literal paths.")
    _aliases([item["rfilename"] for item in siblings])
    prefix = subfolder + "/" if subfolder else ""
    ancestors = {"/".join(prefix.split("/")[:index]) for index in range(len(prefix.split("/")))}
    files: list[RemoteFile] = []
    for item in siblings:
        remote = item["rfilename"]
        if remote.startswith(prefix):
            local = remote[len(prefix):]
            if subfolder and local.split("/")[0].casefold() == "_hf_repository":
                _fail("PATH_ALIAS", "_hf_repository is reserved for ancestor attribution in subfolder acquisitions.")
        elif subfolder and remote.rpartition("/")[0] in ancestors and _attribution(remote.rsplit("/", 1)[-1]):
            local = "_hf_repository/" + remote
        else:
            continue
        suffix = Path(local).suffix.lower()
        if (suffix not in {".onnx", ".data", ".bin", ".qnn", ".json", ".model", ".txt", ".md", ".rst",
                           ".jinja", ".j2", ".tiktoken", ".py"}
                and Path(local).name not in {"Modelfile", ".gitattributes"}
                and not (not suffix and _attribution(Path(local).name))):
            _fail("UNSUPPORTED_FILE", f"Unsupported prebuilt asset (only inert .py source; no runnable/archive assets): "
                  f"{remote}")
        if {".git", ".cache"} & {part.casefold() for part in local.split("/")}:
            _fail("UNSUPPORTED_FILE", f"Administrative/cache directories are not consumer assets: {remote}")
        if re.match(r"(pytorch_model|tf_model|flax_model)", Path(local).name, re.IGNORECASE):
            _fail("UNSUPPORTED_FILE", f"Training weights are outside this prebuilt route: {remote}")
        size = item.get("size")
        limit = MAX_GRAPH if suffix == ".onnx" else MAX_FILE
        if suffix == ".json":
            limit = (64 << 20) if Path(local).name == "tokenizer.json" else MAX_JSON
        elif suffix == ".py":
            limit = MAX_SOURCE
        if type(size) is not int or not 0 <= size <= limit:
            _fail("INVENTORY_LIMIT", f"Missing, invalid or oversized file size: {remote}")
        if not _hex(item.get("blobId"), 40):
            _fail("INVALID_METADATA", f"Missing Git blob identity: {remote}")
        lfs = item.get("lfs")
        if "lfs" in item:
            if (not isinstance(lfs, dict) or not _hex(lfs.get("sha256"), 64)
                    or type(lfs.get("size")) is not int or lfs["size"] != size):
                _fail("INVALID_METADATA", f"Invalid LFS SHA256/size: {remote}")
            algorithm, oid = "sha256", lfs["sha256"]
        else:
            algorithm, oid = "git-blob-sha1", item["blobId"]
        files.append(RemoteFile(remote, local, size, algorithm, oid, item["blobId"]))
    _aliases([file.path for file in files])
    names = {file.path for file in files if file.size > 0}
    if ("genai_config.json" not in names or not names & {"tokenizer.json", "tokenizer.model"}
            or not any(name.endswith(".onnx") for name in names)):
        _fail("MISSING_ASSET", "Selection requires nonempty root genai_config.json, tokenizer and ONNX graph.")
    if sum(file.size for file in files) > MAX_TOTAL:
        _fail("INVENTORY_LIMIT", "Selected inventory exceeds 512 GiB.")
    return tuple(sorted(files, key=lambda file: file.path))


def _policy() -> dict[str, Any]:
    return {"endpoint": ENDPOINT, "public_unauthenticated_only": True, "selection": "whole-subfolder+ancestor-attribution-v1",
            "max_entries": MAX_FILES, "max_depth": MAX_DEPTH, "max_total": MAX_TOTAL, "max_file": MAX_FILE,
            "consumer": dict(CONSUMER), "overwrite": False, "rights_consent_required": True,
            "python_source": {"max_bytes": MAX_SOURCE, "text": "UTF-8 (optional BOM), without NUL",
                             "handling": "Preserve exact inert source bytes; never imported, compiled or executed."},
            "integrity": "Git blob SHA1 includes blob header; LFS content uses SHA256, retaining Git pointer blob ID.",
            "promotion": "windows-same-volume-no-replace", "resume": "verified-whole-files-only-v1"}


@dataclass(frozen=True)
class AcquisitionPlan:
    repo_id: str
    revision: str
    commit: str
    subfolder: str | None
    destination: str
    work_dir: str
    processor_name: str | None
    files: tuple[RemoteFile, ...]

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), "schema_version": 1, "policy": _policy()}

    @property
    def fingerprint(self) -> str:
        return _digest(self.identity())

    def to_dict(self) -> dict[str, Any]:
        return {"action": "plan-acquire-prebuilt", "executed": False, "fingerprint": self.fingerprint,
                "plan": self.identity(), "total_bytes": sum(file.size for file in self.files),
                "rights": "Review model/license/attribution and obtain rights separately. Metadata is not legal approval.",
                "next": "Repeat with --execute --expect-fingerprint HASH --accept-license. "
                        "Python source assets are preserved as bounded inert text bytes, never imported or executed. "
                        "No model/code execution, template inference, installation or publication."}


def _read_state(receipt: Path) -> dict[str, Any]:
    state = read_object(receipt, MAX_STATE)
    expected_keys = {"schema_version", "receipt", "plan", "fingerprint", "stage", "completed", "rights_consent",
                     "partial", "error", "inspection", "state_sha256"}
    if set(state) != expected_keys or state.get("schema_version") != 1:
        _fail("INVALID_RECEIPT", "Unsupported or malformed schema-1 acquisition receipt.")
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    if not _hex(state["state_sha256"], 64) or _digest(unsigned) != state["state_sha256"]:
        _fail("RECEIPT_TAMPER", "Receipt integrity digest changed; no state or assets were modified.")
    if (state["receipt"] != str(receipt) or not isinstance(state["plan"], dict)
            or _digest(state["plan"]) != state["fingerprint"] or not isinstance(state["stage"], str)
            or state["stage"] not in STAGES or state["rights_consent"] is not True):
        _fail("INVALID_RECEIPT", "Receipt path, plan fingerprint or stage is invalid.")
    return state


def plan_acquisition(
    repo_id: str, destination: Path, *, work_dir: Path, revision: str = "main",
    subfolder: str | None = None, processor_name: str | None = None,
    resume: Path | None = None, timeout: float = 30,
) -> AcquisitionPlan:
    """Metadata-only plan. Resume pins the receipt commit, not the moving branch."""
    _safe_input(repo_id, revision, subfolder)
    deadline = _deadline(timeout)
    try:
        work_dir = regular_path(work_dir, directory=True)
        saved = None
        if resume is not None:
            resume = regular_path(resume)
            saved = _read_state(resume)
            prior = saved["plan"]
            if (resume.name != "receipt.json" or resume.parent.parent != work_dir
                    or not re.fullmatch(r"hf-acquire-[0-9a-f]{32}", resume.parent.name)):
                _fail("INVALID_RECEIPT", "Resume requires the original receipt under the explicit owned work parent.")
            destination = Path(os.path.abspath(destination.expanduser()))
            options = {"repo_id": repo_id, "revision": revision, "subfolder": subfolder,
                       "destination": str(destination), "work_dir": str(work_dir), "processor_name": processor_name}
            if any(prior.get(key) != value for key, value in options.items()) or not _hex(prior.get("commit"), 40):
                _fail("PLAN_CHANGED", "Resume options differ from the receipt's immutable plan.")
            commit = prior["commit"]
            regular_path(destination.parent, directory=True)
        else:
            destination = _absent_destination(destination)
            commit = _metadata(repo_id, revision, deadline)["sha"]
        if processor_name is not None and (not processor_name.strip() or len(processor_name) > 512):
            _fail("INVALID_TARGET", "Processor comparison must be a nonempty bounded exact name.")
        metadata = _metadata(repo_id, commit, deadline)
        if metadata["sha"] != commit:
            _fail("METADATA_DRIFT", "Immutable commit resolution changed.")
        files = _files(metadata, subfolder)
        plan = AcquisitionPlan(repo_id, revision, commit, subfolder, str(destination), str(work_dir),
                               processor_name, files)
        if any(max(len(str(destination / file.path)), len(str(work_dir / ("x" * 43) / "bundle" / file.path))) > 240
               for file in files):
            _fail("PATH_LIMIT", "Acquired paths exceed the conservative 240-character Windows path bound.")
        if saved is not None and (saved["plan"] != json.loads(_encoded(plan.identity()))
                                  or saved["fingerprint"] != plan.fingerprint):
            _fail("METADATA_DRIFT", "Receipt policy/inventory differs from the current immutable metadata.")
        return plan
    except OSError as exc:
        raise NpuModelError("acquire", "LOCAL_IO_FAILED", "Cannot validate acquisition paths.", cause=exc) from exc
    except KeyboardInterrupt as exc:
        raise NpuModelError("acquire", "CANCELLED", "Metadata planning interrupted; no data acquired.", cause=exc) from exc


def _write_state(receipt: Path, state: dict[str, Any], previous: str | None) -> None:
    regular_path(receipt.parent, directory=True)
    if previous is None:
        if os.path.lexists(receipt):
            _fail("RECEIPT_TAMPER", "New receipt unexpectedly exists.")
    elif _read_state(receipt)["state_sha256"] != previous:
        _fail("RECEIPT_TAMPER", "Receipt changed during execution.")
    state = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = _digest(state)
    data = _encoded(state)
    if len(data) > MAX_STATE:
        _fail("RECEIPT_LIMIT", "Receipt exceeds 16 MiB; no completion is claimed.")
    pending = receipt.parent / "receipt.pending"
    stream = pending.open("xb")
    try:
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, receipt)
    finally:
        pending.unlink(missing_ok=True)


def _save(receipt: Path, state: dict[str, Any]) -> None:
    intended = _digest({key: value for key, value in state.items() if key != "state_sha256"})
    try:
        _write_state(receipt, state, state.get("state_sha256"))
        if _read_state(receipt)["state_sha256"] != intended:
            _fail("RECEIPT_TAMPER", "Receipt changed immediately after its atomic update.")
    except (OSError, KeyboardInterrupt):
        # Atomic replacement can finish before Python receives an interruption.
        if os.path.lexists(receipt) and _read_state(receipt)["state_sha256"] == intended:
            state["state_sha256"] = intended
        raise
    state["state_sha256"] = intended


def _local_hash(path: Path, deadline: float) -> tuple[int, str, str]:
    regular_path(path)
    before = path.stat()
    if before.st_size > MAX_FILE:
        _fail("STAGED_TAMPER", "Local file exceeds the bounded acquisition file size.")
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before.st_size}\0".encode(), usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            _remaining(deadline)
            size += len(chunk)
            if size > before.st_size:
                _fail("STAGED_TAMPER", "Local file grew while hashing.")
            sha256.update(chunk)
            blob.update(chunk)
    after = regular_path(path).stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        _fail("STAGED_TAMPER", "Staged file changed while hashing.")
    return size, sha256.hexdigest(), blob.hexdigest()


def _verified(path: Path, file: RemoteFile, deadline: float) -> dict[str, Any]:
    if regular_path(path).stat().st_size != file.size:
        _fail("CHECKSUM_MISMATCH", f"Size differs from immutable inventory: {file.path}")
    size, sha256, blob = _local_hash(path, deadline)
    if size != file.size or (sha256 if file.algorithm == "sha256" else blob) != file.oid:
        _fail("CHECKSUM_MISMATCH", f"Size or immutable {file.algorithm} mismatch: {file.path}")
    if Path(file.path).suffix.lower() == ".py":
        if size > MAX_SOURCE:
            _fail("INVENTORY_LIMIT", f"Python source exceeds the bounded text policy: {file.path}")
        with regular_path(path).open("rb") as stream:
            data = stream.read(MAX_SOURCE + 1)
        if len(data) > MAX_SOURCE:
            _fail("INVENTORY_LIMIT", f"Python source exceeds the bounded text policy: {file.path}")
        try:
            data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise NpuModelError("acquire", "UNSUPPORTED_SOURCE", "Python source must be inert UTF-8 text.",
                                cause=exc) from exc
        if b"\0" in data:
            _fail("UNSUPPORTED_SOURCE", f"Python source must be text without NUL bytes: {file.path}")
    return {"path": file.path, "size": size, "sha256": sha256}


def _transfer(
    plan: AcquisitionPlan, file: RemoteFile, partial: Path, deadline: float, progress: dict[str, Any],
) -> None:
    url = f"{ENDPOINT}/{plan.repo_id}/resolve/{plan.commit}/{quote(file.remote_path, safe='/')}?download=true"
    with _response(url, deadline, redirects=True) as response:
        length = response.getheader("Content-Length")
        if length is not None and length != str(file.size):
            _fail("METADATA_DRIFT", f"Content-Length differs from the immutable inventory: {file.path}")
        with partial.open("xb") as stream:
            size = 0
            digest = hashlib.sha256()
            while True:
                _remaining(deadline)
                chunk = response.read1(min(CHUNK, file.size - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > file.size:
                    _fail("CHECKSUM_MISMATCH", f"Transfer exceeds planned size: {file.path}")
                stream.write(chunk)
                digest.update(chunk)
                progress.update(size=size, sha256=digest.hexdigest())
            stream.flush()
            os.fsync(stream.fileno())


def _layout(session: Path, plan: AcquisitionPlan, state: dict[str, Any], deadline: float) -> Path:
    if state["plan"] != json.loads(_encoded(plan.identity())) or state["fingerprint"] != plan.fingerprint:
        _fail("PLAN_CHANGED", "Receipt changed after planning.")
    if {item.name for item in session.iterdir()} - {"receipt.json", "bundle", "partial", "lock"}:
        _fail("STAGED_TAMPER", "Unexpected files in acquisition work session.")
    files = {file.path: file for file in plan.files}
    completed = state["completed"]
    if not isinstance(completed, list) or len(completed) > len(files):
        _fail("INVALID_RECEIPT", "Invalid completed-file inventory.")
    seen: set[str] = set()
    for item in completed:
        if (not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}
                or not isinstance(item["path"], str) or item["path"] not in files
                or item["path"] in seen or type(item["size"]) is not int or item["size"] != files[item["path"]].size
                or not _hex(item["sha256"], 64)):
            _fail("INVALID_RECEIPT", "Malformed or duplicate completed-file entry.")
        seen.add(item["path"])
    destination = Path(plan.destination)
    if state["stage"] == "completed" and not os.path.lexists(destination):
        _fail("DESTINATION_CHANGED", "Completed destination is missing.")
    promoted = state["stage"] in {"promoting", "completed"} and os.path.lexists(destination)
    root = destination if promoted else session / "bundle"
    if promoted and os.path.lexists(session / "bundle"):
        _fail("DESTINATION_EXISTS", "Both staged bundle and destination exist; refusing ambiguous recovery.")
    if not promoted:
        _absent_destination(destination)
    regular_path(root, directory=True)
    expected_dirs = {str(Path(file.path).parent).replace("\\", "/") for file in plan.files}
    expected_dirs |= {"/".join(file.path.split("/")[:index]) for file in plan.files
                      for index in range(1, len(file.path.split("/")))}
    for parent, dirs, _ in os.walk(root, followlinks=False):
        for name in dirs:
            child = regular_path(Path(parent) / name, directory=True)
            if child.relative_to(root).as_posix() not in expected_dirs:
                _fail("STAGED_TAMPER", "Unexpected staged directory.")
    assets = inventory(root)
    if {asset.path for asset in assets} != seen:
        _fail("STAGED_TAMPER", "Staged inventory differs from verified completed files.")
    for item in completed:
        actual = _verified(root / item["path"], files[item["path"]], deadline)
        if actual != item:
            _fail("STAGED_TAMPER", "Staged content SHA256 differs from the receipt.")
    partial = session / "partial"
    recorded = state["partial"]
    if recorded is not None:
        if (not isinstance(recorded, dict) or set(recorded) != {"path", "size", "sha256"}
                or not isinstance(recorded["path"], str) or recorded["path"] not in files
                or recorded["path"] in seen or type(recorded["size"]) is not int
                or not 0 <= recorded["size"] <= files[recorded["path"]].size or not _hex(recorded["sha256"], 64)):
            _fail("INVALID_RECEIPT", "Malformed interrupted-file entry.")
        size, sha256, _ = _local_hash(partial, deadline)
        if size != recorded["size"] or sha256 != recorded["sha256"]:
            _fail("STAGED_TAMPER", "Interrupted bytes differ from the receipt; nothing was discarded.")
    elif os.path.lexists(partial):
        _fail("STAGED_TAMPER", "Unrecorded partial file; unsafe to resume.")
    if promoted and (len(seen) != len(files) or recorded is not None):
        _fail("INVALID_RECEIPT", "Promotion requires every file to be verified.")
    return root


def execute_acquisition(
    plan: AcquisitionPlan, *, expected_fingerprint: str | None, accept_license: bool,
    resume: Path | None = None, timeout: float = 300,
) -> dict[str, Any]:
    """Acquire only; resume is file-level and never trusts a cache or partial bytes."""
    if expected_fingerprint != plan.fingerprint:
        _fail("PLAN_CHANGED", "Supply the exact reviewed --expect-fingerprint; no transfer was attempted.")
    if not accept_license:
        _fail("RIGHTS_CONSENT_REQUIRED", "Review applicable licenses/attribution and obtain download/use rights; "
              "--accept-license records caller consent, not legal approval.")
    deadline = _deadline(timeout)
    if os.name != "nt":
        _fail("ATOMIC_PROMOTION_UNAVAILABLE", "This bounded route requires Windows no-replace directory rename.")
    try:
        fresh = plan_acquisition(plan.repo_id, Path(plan.destination), work_dir=Path(plan.work_dir),
                                 revision=plan.revision if resume is not None else plan.commit,
                                 subfolder=plan.subfolder, processor_name=plan.processor_name,
                                 resume=resume, timeout=max(1, _remaining(deadline)))
        # Resume uses the original mutable revision as an option, but always reads the pinned commit.
        if resume is None:
            fresh = AcquisitionPlan(fresh.repo_id, plan.revision, fresh.commit, fresh.subfolder, fresh.destination,
                                    fresh.work_dir, fresh.processor_name, fresh.files)
        if fresh.fingerprint != plan.fingerprint:
            _fail("METADATA_DRIFT", "Acquisition plan changed before transfer.")
        destination = Path(plan.destination)
        if destination.parent.stat().st_dev != Path(plan.work_dir).stat().st_dev:
            _fail("CROSS_VOLUME", "Work parent and destination must share a volume for atomic promotion.")
        if resume is None:
            session = Path(plan.work_dir) / ("hf-acquire-" + uuid.uuid4().hex)
            session.mkdir()
            (session / "bundle").mkdir()
            receipt = session / "receipt.json"
            state: dict[str, Any] = {"schema_version": 1, "receipt": str(receipt),
                                     "plan": json.loads(_encoded(plan.identity())), "fingerprint": plan.fingerprint,
                                     "stage": "downloading", "completed": [], "partial": None, "rights_consent": True,
                                     "error": None, "inspection": None}
            _save(receipt, state)
        else:
            receipt = regular_path(resume)
            session = receipt.parent
            state = _read_state(receipt)
        lock = session / "lock"
        with lock.open("xb"):
            pass
        active: RemoteFile | None = None
        progress: dict[str, Any] = {}
        try:
            root = _layout(session, plan, state, deadline)
            # Validation failures above do not rewrite a suspect receipt.
            try:
                if state["partial"] is not None:
                    (session / "partial").unlink()
                    state["partial"] = None
                state.update(stage="downloading" if root != destination else "promoting", error=None)
                _save(receipt, state)
                completed = {item["path"] for item in state["completed"]}
                for file in plan.files:
                    if file.path in completed:
                        continue
                    active = file
                    progress = {"path": file.path, "size": 0, "sha256": hashlib.sha256(b"").hexdigest()}
                    partial = session / "partial"
                    _transfer(plan, file, partial, deadline, progress)
                    verified = _verified(partial, file, deadline)
                    target = root / file.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    regular_path(target.parent, directory=True)
                    try:
                        partial.rename(target)
                    finally:
                        # A caught interruption immediately after rename must not orphan a verified whole file.
                        if not os.path.lexists(partial) and os.path.lexists(target):
                            state["completed"].append(verified)
                            state["completed"].sort(key=lambda item: item["path"])
                    _save(receipt, state)
                state["stage"] = "validating" if root != destination else "promoting"
                _save(receipt, state)
                report = inspect_prebuilt(root, processor_name=plan.processor_name)
                if not report.importable:
                    _fail("STRUCTURAL_MISMATCH", "Acquired folder failed inspect-prebuilt: "
                          + ", ".join(error["code"] for error in report.errors)
                          + f"; target={report.target['status']}.")
                _remaining(deadline)
                metadata = _metadata(plan.repo_id, plan.commit, deadline)
                if metadata["sha"] != plan.commit or _files(metadata, plan.subfolder) != plan.files:
                    _fail("METADATA_DRIFT", "Immutable inventory changed during acquisition.")
                state["inspection"] = report.to_dict()
                # Rehash after inspection/metadata, immediately before promotion.
                _layout(session, plan, state, deadline)
                state["stage"] = "promoting"
                _save(receipt, state)
                if root != destination:
                    _absent_destination(destination)
                    regular_path(root, directory=True)
                    root.rename(destination)  # Windows rename refuses even an existing empty destination.
                state["inspection"]["input"] = str(destination)
                state["stage"] = "completed"
                _save(receipt, state)
                return {"action": "acquire-prebuilt", "executed": True, "stage": "completed",
                        "destination": str(destination), "receipt": str(receipt),
                        "fingerprint": plan.fingerprint, "inspection": state["inspection"],
                        "rights": "Caller consent recorded; no license/provenance/authenticity certification."}
            except (NpuModelError, OSError, KeyboardInterrupt) as exc:
                # A caught interruption can record its partial bytes. An unclean
                # process kill leaving unrecorded bytes deliberately fails closed.
                code = exc.reason_code if isinstance(exc, NpuModelError) else (
                    "CANCELLED" if isinstance(exc, KeyboardInterrupt) else "LOCAL_IO_FAILED")
                partial = session / "partial"
                if os.path.lexists(partial):
                    if active is None:
                        _fail("STAGED_TAMPER", "Unexpected partial file outside an active transfer.")
                    regular_path(partial)
                    state["partial"] = progress
                else:
                    state["partial"] = None
                stage = "promoting" if not os.path.lexists(session / "bundle") else (
                    "interrupted" if code == "CANCELLED" else "failed")
                state.update(stage=stage,
                             error={"code": code, "message": "Acquisition incomplete; review before explicit resume."})
                _save(receipt, state)
                raise NpuModelError("acquire", code, f"Acquisition incomplete. Receipt: {receipt}", cause=exc) from exc
        finally:
            lock.unlink()
    except OSError as exc:
        raise NpuModelError("acquire", "LOCAL_IO_FAILED", "Acquisition filesystem operation failed; "
                            "existing destination was not removed.", cause=exc) from exc
