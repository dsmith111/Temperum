"""Explicit local Ollama create, from a hash-bound inspected folder snapshot."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import sha256_file
from npu_model.bundle.inspect import Inspection, inspect_prebuilt, inventory, regular_path

MAX_MODELFILE = 1 << 20
MAX_RESPONSE = 4 << 20
TAGS_TIMEOUT = 180.0


def _fail(code: str, message: str) -> NoReturn:
    raise NpuModelError(stage="import", reason_code=code, message=message)


def local_endpoint(host: str | None) -> str:
    if not host:
        _fail("HOST_REQUIRED", "Specify --host or OLLAMA_HOST; no default server is selected or started.")
    try:
        url = urlsplit(str(host))
        port = url.port
    except ValueError as exc:
        _fail("INVALID_HOST", f"Invalid local endpoint: {exc}")
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"} or port is None
            or not 0 < port <= 65535 or url.username is not None or url.password is not None
            or url.path not in {"", "/"} or url.query or url.fragment
            or "?" in str(host) or "#" in str(host)):
        _fail("LOCAL_HOST_ONLY", "Use explicit http://127.0.0.1:PORT or http://[::1]:PORT without credentials.")
    return f"http://{'[::1]' if url.hostname == '::1' else '127.0.0.1'}:{port}"


def _model_name(name: str) -> str:
    # Bounded local tags, not a remote registry hostname or URL.
    component = r"[a-z0-9][a-z0-9_.-]*"
    if len(name) > 128 or not re.fullmatch(rf"(?:{component}/)?{component}(?::{component})?", name):
        _fail("INVALID_NAME", "Use a lowercase local model tag, optionally namespace/name:tag (no registry URL).")
    return name if ":" in name else name + ":latest"


def _canonical_tag(name: str) -> str:
    name = name.removeprefix("registry.ollama.ai/").removeprefix("library/")
    return name if ":" in name else name + ":latest"


@dataclass(frozen=True)
class Modelfile:
    text: str
    source: str
    sha256: str
    from_span: tuple[int, int]

    def render(self, bundle: Path) -> str:
        # Go's Modelfile parser treats backslashes literally inside quotes.
        source = str(bundle)
        if any(char in source for char in '"\r\n'):
            _fail("INVALID_WORK_PATH", "Working path cannot contain quotes or newlines.")
        start, end = self.from_span
        return self.text[:start] + f'FROM "{source}"' + self.text[end:]


def read_modelfile(path: Path, bundle: Path, context_length: int | None) -> Modelfile:
    path = regular_path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_MODELFILE + 1)
    if not data or len(data) > MAX_MODELFILE:
        _fail("INVALID_MODELFILE", "Modelfile must be nonempty and at most 1 MiB.")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        _fail("INVALID_MODELFILE", f"Modelfile must be UTF-8: {exc}")
    if "\x00" in text or text.startswith("\ufeff"):
        _fail("INVALID_MODELFILE", "Modelfile must not contain NUL or a BOM.")
    commands: list[tuple[str, str, tuple[int, int]]] = []
    offset = 0
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        start = offset
        index += 1
        offset += len(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z]+)[ \t]+(.+)", stripped)
        if match is None:
            _fail("INVALID_MODELFILE", f"Unsupported Modelfile syntax: {stripped[:120]}")
        command, arg = match.group(1).upper(), match.group(2)
        if command not in {"FROM", "TEMPLATE", "SYSTEM", "LICENSE", "PARAMETER", "MESSAGE"}:
            _fail("UNSAFE_MODELFILE", f"{command} is outside the bounded local import Modelfile contract.")
        if arg.startswith('"""'):
            if command in {"FROM", "PARAMETER", "MESSAGE"}:
                _fail("INVALID_MODELFILE", f"{command} cannot use a multiline argument in this route.")
            pieces = [arg[3:]]
            while '"""' not in pieces[-1]:
                if index >= len(lines):
                    _fail("INVALID_MODELFILE", "Unterminated triple-quoted Modelfile argument.")
                next_line = lines[index]
                index += 1
                offset += len(next_line)
                pieces.append(next_line)
            raw = "\n".join(pieces)
            arg, tail = raw.split('"""', 1)
            if tail.strip():
                _fail("INVALID_MODELFILE", "Trailing content after multiline Modelfile argument.")
        elif arg.startswith('"'):
            if len(arg) < 2 or not arg.endswith('"') or '"' in arg[1:-1]:
                _fail("INVALID_MODELFILE", "Malformed quoted Modelfile argument.")
            arg = arg[1:-1]
        commands.append((command, arg, (start, offset - len(lines[index - 1]) + len(lines[index - 1].rstrip("\r\n")))))
    sources = [entry for entry in commands if entry[0] == "FROM"]
    templates = [entry[1] for entry in commands if entry[0] == "TEMPLATE"]
    if len(sources) != 1:
        _fail("LOCAL_FROM_REQUIRED", "Modelfile requires exactly one FROM pointing to the inspected local folder.")
    source = sources[0][1]
    if not source or any(char in source for char in "*?[]\r\n") or source.startswith("~"):
        _fail("LOCAL_FROM_REQUIRED", "FROM must be one literal local directory (no tag, glob, URL or expansion).")
    resolved = Path(source)
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    try:
        resolved = regular_path(resolved, directory=True)
    except (OSError, NpuModelError) as exc:
        _fail("LOCAL_FROM_REQUIRED", f"FROM must exist locally and equal the inspected folder: {exc}")
    if resolved != bundle:
        _fail("SOURCE_SUBSTITUTION", "Modelfile FROM does not point to the inspected local folder.")
    if (len(templates) != 1 or not templates[0].strip() or "{%" in templates[0] or "{#" in templates[0]
            or not re.search(r"\{\{[^{}]*\.(Prompt|Messages|System)\b[^{}]*\}\}", templates[0])):
        _fail("OLLAMA_TEMPLATE_REQUIRED", "Supply one known-correct Ollama Go TEMPLATE using .Prompt/.Messages/"
              ".System; bundle Jinja is not an Ollama template. Role tokens are never inferred.")
    parameters = [entry[1].split() for entry in commands if entry[0] == "PARAMETER"]
    if any(len(parameter) < 2 for parameter in parameters):
        _fail("INVALID_MODELFILE", "PARAMETER requires a name and value.")
    contexts = [parameter[1:] for parameter in parameters if parameter[0].lower() == "num_ctx"]
    if contexts:
        if (len(contexts) != 1 or len(contexts[0]) != 1 or not re.fullmatch(r"[0-9]{1,10}", contexts[0][0])
                or int(contexts[0][0]) <= 0 or context_length is None
                or int(contexts[0][0]) > context_length):
            _fail("CONTEXT_CONFLICT", "PARAMETER num_ctx must be positive and not exceed the inspected context.")
    return Modelfile(text, str(path), hashlib.sha256(data).hexdigest(), sources[0][2])


@dataclass(frozen=True)
class ImportPlan:
    inspection: Inspection
    name: str
    endpoint: str
    executable: str
    executable_sha256: str
    modelfile: Modelfile
    overwrite: bool
    fingerprint: str
    processor_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"action": "plan-local-create", "executed": False, "fingerprint": self.fingerprint,
                "name": self.name, "host": self.endpoint, "ollama_exe": self.executable,
                "ollama_exe_sha256": self.executable_sha256, "overwrite": self.overwrite,
                "modelfile": {"path": self.modelfile.source, "sha256": self.modelfile.sha256,
                              "template_evidence": "caller-supplied; semantic correctness not certified",
                              "preservation": "Only FROM is rebound to an exact local snapshot."},
                "inspection": self.inspection.to_dict(),
                "next": "Review plan; repeat with --execute --expect-fingerprint FINGERPRINT. "
                        "A running local compatible fork is required. No server is started."}


def plan_import(
    bundle: Path, name: str, *, executable: Path | None, host: str | None,
    modelfile: Path | None = None, overwrite: bool = False, processor_name: str | None = None,
) -> ImportPlan:
    endpoint = local_endpoint(host or os.environ.get("OLLAMA_HOST"))
    name = _model_name(name)
    if executable is None:
        setting = os.environ.get("NPU_MODEL_OLLAMA_EXE")
        if not setting:
            _fail("EXE_REQUIRED", "Specify --ollama-exe or NPU_MODEL_OLLAMA_EXE (compatible fork); no PATH default.")
        executable = Path(setting)
    try:
        executable = regular_path(executable)
        if executable.suffix.lower() in {".bat", ".cmd", ".ps1"}:
            _fail("INVALID_EXE", "Use an executable, not a shell script.")
        if (executable.parent / "ollama app.exe").exists():
            _fail("AUTO_START_RISK", "Use the standalone retained fork executable, not an app installation.")
        executable_hash = sha256_file(executable)
    except OSError as exc:
        _fail("INVALID_EXE", f"Cannot read local Ollama executable: {exc}")
    inspection = inspect_prebuilt(bundle, processor_name=processor_name)
    if not inspection.importable:
        codes = ", ".join(error["code"] for error in inspection.errors)
        _fail("INSPECTION_FAILED", f"Prebuilt input is not structurally importable ({codes or inspection.target['status']}). "
              "Run inspect-prebuilt --input PATH for the complete machine-readable report.")
    bundle = Path(inspection.input)
    try:
        parsed = read_modelfile(modelfile or bundle / "Modelfile", bundle, inspection.context_length)
    except OSError as exc:
        _fail("MODELFILE_REQUIRED", f"Supply --modelfile with a known-correct local FROM and Go TEMPLATE: {exc}")
    identity = {"content": inspection.content_fingerprint, "modelfile": parsed.sha256,
                "exe": executable_hash, "name": name, "host": endpoint, "overwrite": overwrite,
                "processor_name": processor_name}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return ImportPlan(inspection, name, endpoint, str(executable), executable_hash, parsed, overwrite,
                      fingerprint, processor_name)


def _api(endpoint: str, path: str, timeout: float, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(str(url.hostname), url.port, timeout=timeout)
    try:
        # http.client never uses proxies, follows redirects, or sends authorization.
        if body is None:
            connection.request("GET", path, headers={"Accept": "application/json"})
        else:
            connection.request("POST", path, body=json.dumps(body).encode("utf-8"),
                               headers={"Accept": "application/json", "Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read(MAX_RESPONSE + 1)
        if response.status != 200:
            _fail("LOCAL_SERVER_ERROR", f"Local Ollama {path} returned HTTP {response.status}; no redirects/auth retry.")
        if len(data) > MAX_RESPONSE:
            _fail("LOCAL_SERVER_ERROR", "Local Ollama response exceeds 4 MiB.")
        value = json.loads(data)
        if not isinstance(value, dict):
            _fail("LOCAL_SERVER_ERROR", "Local Ollama response must be a JSON object.")
        return cast(dict[str, Any], value)
    except (OSError, http.client.HTTPException, ValueError, RecursionError) as exc:
        _fail("LOCAL_SERVER_UNAVAILABLE", f"Cannot query existing local fork: {exc}. Start it separately; import never starts it.")
    finally:
        connection.close()


def _tags(endpoint: str, timeout: float) -> set[str]:
    value = _api(endpoint, "/api/tags", timeout).get("models")
    if not isinstance(value, list) or len(value) > 4096:
        _fail("LOCAL_SERVER_ERROR", "Local Ollama tags response must contain a bounded models array.")
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            _fail("LOCAL_SERVER_ERROR", "Malformed local Ollama model tag.")
        names.add(_canonical_tag(item["name"]))
    return names


def _child_env(executable: Path, workspace: Path, endpoint: str) -> dict[str, str]:
    # Do not propagate tokens, auth, proxy settings, DLL search paths or the user's home.
    safe = {"SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE", "PATHEXT"}
    env = {key: value for key, value in os.environ.items() if key.upper() in safe}
    system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    env.update({"OLLAMA_HOST": endpoint, "OLLAMA_AUTH": "false", "OLLAMA_NO_CLOUD": "1",
                "NO_PROXY": "*", "no_proxy": "*", "PATH": os.pathsep.join((str(executable.parent), str(system))),
                "HOME": str(workspace), "USERPROFILE": str(workspace), "LOCALAPPDATA": str(workspace),
                "APPDATA": str(workspace)})
    return env


def _run_create(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> None:
    # Stream output to avoid unbounded capture from a verbose native CLI.
    try:
        result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr,
                                timeout=timeout, check=False, shell=False)
    except subprocess.TimeoutExpired as exc:
        _fail("CREATE_TIMEOUT", f"Ollama create exceeded {timeout:.0f}s; tag state may be partial. "
              f"Inspect the dedicated local server before retrying: {exc}")
    except OSError as exc:
        _fail("CREATE_LAUNCH_FAILED", f"Cannot launch the selected fork executable: {exc}")
    if result.returncode != 0:
        _fail("CREATE_FAILED", f"Ollama create returned {result.returncode}; inspect CLI/server output. "
              "No install success is claimed; the local server may contain partial blobs.")


def execute_import(plan: ImportPlan, *, expected_fingerprint: str | None, work_dir: Path,
                   timeout: float = 300) -> dict[str, Any]:
    if not expected_fingerprint or expected_fingerprint != plan.fingerprint:
        _fail("PLAN_CHANGED", "Execution requires --expect-fingerprint from the reviewed import plan; "
              "bundle, Modelfile, executable, endpoint, name or overwrite policy changed.")
    if not 1 <= timeout <= 3600:
        _fail("INVALID_TIMEOUT", "Import timeout must be between 1 and 3600 seconds.")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        duration = deadline - time.monotonic()
        if duration <= 0:
            _fail("CREATE_TIMEOUT", "Import time budget exhausted; no install success is claimed.")
        return duration

    # Reject overwrite before any create subprocess or staging writes.
    if _canonical_tag(plan.name) in _tags(plan.endpoint, remaining()) and not plan.overwrite:
        _fail("TAG_EXISTS", f"Local tag {plan.name} exists. Review a new plan with --overwrite to replace it.")
    try:
        work_dir = regular_path(work_dir, directory=True)
        source = Path(plan.inspection.input)
        if work_dir == source or work_dir.is_relative_to(source):
            _fail("INVALID_WORK_PATH", "Work directory must be outside the immutable input folder.")
        workspace = work_dir / f"local-import-{uuid.uuid4().hex}"
        workspace.mkdir()
    except OSError as exc:
        _fail("INVALID_WORK_PATH", f"Provide an existing regular --work-dir outside the input: {exc}")
    try:
        snapshot = workspace / "bundle"
        snapshot.mkdir()
        for asset in plan.inspection.assets:
            remaining()
            src = regular_path(source.joinpath(*asset.path.split("/")))
            dest = snapshot.joinpath(*asset.path.split("/"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        if inventory(snapshot) != plan.inspection.assets or inventory(source) != plan.inspection.assets:
            _fail("ASSET_CHANGED", "Input changed since inspection or while snapshotting; create was not invoked.")
        remaining()
        snapshot_report = inspect_prebuilt(snapshot, processor_name=plan.processor_name)
        if not snapshot_report.importable or snapshot_report.content_fingerprint != plan.inspection.content_fingerprint:
            _fail("ASSET_CHANGED", "Snapshot failed repeat structural/declaration validation; create was not invoked.")
        if (sha256_file(regular_path(Path(plan.executable))) != plan.executable_sha256
                or sha256_file(regular_path(Path(plan.modelfile.source))) != plan.modelfile.sha256):
            _fail("PLAN_CHANGED", "Executable or Modelfile changed since planning; create was not invoked.")
        if (Path(plan.executable).parent / "ollama app.exe").exists():
            _fail("AUTO_START_RISK", "An adjacent Ollama app appeared; use a standalone retained fork.")
        generated = workspace / "Modelfile"
        generated.write_text(plan.modelfile.render(snapshot), encoding="utf-8", newline="")
        # Check again as late as possible; simultaneous independent tag writers are not locked.
        if _canonical_tag(plan.name) in _tags(plan.endpoint, remaining()) and not plan.overwrite:
            _fail("TAG_EXISTS", "Tag appeared before create; create was not invoked.")
        _run_create([plan.executable, "create", plan.name, "-f", str(generated)], workspace,
                    _child_env(Path(plan.executable), workspace, plan.endpoint), remaining())
        try:
            installed = _tags(plan.endpoint, remaining())
        except NpuModelError as exc:
            raise NpuModelError("import", "CREATE_NOT_CONFIRMED",
                                "CLI returned zero but local tag confirmation failed. The tag may already exist; "
                                "inspect the dedicated endpoint before retrying. No automatic overwrite or success.",
                                cause=exc) from exc
        if _canonical_tag(plan.name) not in installed:
            _fail("CREATE_NOT_CONFIRMED", "CLI returned zero but the dedicated endpoint does not list the created tag.")
    except OSError as exc:
        _fail("IMPORT_IO_FAILED", f"Local snapshot/import I/O failed: {exc}")
    finally:
        active_error = sys.exc_info()[1]
        try:
            shutil.rmtree(workspace)
        except OSError as exc:
            message = f"Cannot remove owned staging folder {workspace}: {exc}. Remove it manually; no ACL changes made."
            if isinstance(active_error, NpuModelError):
                active_error.hint = message
            else:
                _fail("STAGING_CLEANUP_FAILED", message + " Check the local server's tag state before retrying.")
    return {"action": "local-create", "executed": True, "installed": True, "name": plan.name,
            "host": plan.endpoint, "fingerprint": plan.fingerprint,
            "content_fingerprint": plan.inspection.content_fingerprint,
            "native_evidence": {"status": "not-run", "certified": False},
            "detail": "Local create returned zero and tag presence was confirmed; generation/native execution unverified."}


def install_gguf(bundle: Path, name: str, *, executable: Path, host: str | None, work_dir: Path,
                 overwrite: bool = False, timeout: float = 900) -> dict[str, Any]:
    """`ollama create` from a GGUF bundle's Modelfile, through an already running local fork.

    The GGUF carries its own chat template; the Modelfile adds a checked Ollama template when the
    converter recognizes the format (gguf.modelfile). Afterwards the fork is asked which template the
    model got, because `ollama create` substitutes a built-in one for formats close to it.
    """
    endpoint = local_endpoint(host)
    tag = _model_name(name)
    modelfile = bundle / "Modelfile"
    if not modelfile.is_file() or not (bundle / "model.gguf").is_file():
        _fail("BUNDLE_INVALID", f"{bundle} is not a GGUF bundle (needs model.gguf and Modelfile).")
    if not executable.is_file():
        _fail("EXE_REQUIRED", f"Fork executable not found: {executable}")
    # A fork lists large models slowly the first time (33 s for a 3 GB ORT bundle).
    if _canonical_tag(tag) in _tags(endpoint, TAGS_TIMEOUT) and not overwrite:
        _fail("TAG_EXISTS", f"Local tag {tag} exists. Pass --overwrite to replace it.")
    work_dir.mkdir(parents=True, exist_ok=True)
    _run_create([str(executable), "create", tag, "-f", str(modelfile)], bundle,
                _child_env(executable, work_dir, endpoint), timeout)
    if _canonical_tag(tag) not in _tags(endpoint, TAGS_TIMEOUT):
        _fail("CREATE_FAILED", f"Ollama create finished but {tag} is not listed by {endpoint}.")
    result: dict[str, Any] = {"name": tag, "endpoint": endpoint}
    try:
        shown = str(_api(endpoint, "/api/show", TAGS_TIMEOUT, {"model": tag}).get("template") or "")
    except NpuModelError:
        shown = None
    result["template"] = template_source(modelfile.read_text(encoding="utf-8"), shown)
    if result["template"] == "fork built-in":
        result["warning"] = ("the fork replaced the model's chat template with a similar built-in one, which the "
                             "converter did not check; chats may be formatted differently")
    return result


def template_source(modelfile_text: str, shown: str | None) -> str:
    """Which chat template the fork renders for a GGUF model (`ollama show`'s template field).

    The fork shows the template it will use: the GGUF's own (Jinja, rendered by llama-server) when
    that one supports more, such as tool calls, else the Go template stored with the model.
    """
    if shown is None:
        return "unknown"
    if "{%" in shown:
        return "gguf"  # the model's own template
    if not shown:
        return "none"
    if re.search(r"^TEMPLATE ", modelfile_text, re.MULTILINE):
        return "bundle"  # the converter's checked template
    return "fork built-in"
