"""Read-only, bounded inspection of a local consumer-ready ORT GenAI folder."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from temperum.core.errors import TemperumError
from temperum.core.manifest import sha256_file
from temperum.bundle.genai_config import validate_genai_config

MAX_FILES = 4096
MAX_DEPTH = 32
MAX_JSON = 4 << 20
MAX_GRAPH = 256 << 20
MAX_TOTAL = 512 << 30
CONSUMER = {"platform": "windows/arm64", "genai_version": "0.12.2", "ort_version": "1.24.1"}


def fail(code: str, message: str) -> NoReturn:
    raise TemperumError(stage="prebuilt", reason_code=code, message=message)


def regular_path(path: Path, *, directory: bool = False) -> Path:
    """Check ancestors before resolving, including Windows junction/reparse points."""
    path = Path(os.path.abspath(path.expanduser()))
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            fail("SYMLINK_ASSET", f"Symlink/reparse point is not allowed: {part}")
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        fail("INVALID_ASSET", f"Expected a regular {'directory' if directory else 'file'}: {path}")
    return path


def relative_name(name: Any) -> str:
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 512:
        fail("INVALID_REFERENCE", f"Invalid relative asset reference: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts) or "\\" in name:
        fail("UNSAFE_REFERENCE", f"Asset reference must be a canonical contained relative path: {name!r}")
    for part in parts:
        base = part.split(".")[0].upper()
        reserved = base in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        reserved |= bool(re.fullmatch(r"(COM|LPT)[1-9¹²³]", base))
        if (reserved or part.rstrip(". ") != part or any(c in '<>:"|?*' for c in part)
                or any(unicodedata.category(c) == "Cc" for c in part)):
            fail("UNSAFE_REFERENCE", f"Nonportable asset path: {name!r}")
    return cast(str, name)


@dataclass(frozen=True)
class Asset:
    path: str
    size: int
    sha256: str


def inventory(root: Path) -> tuple[Asset, ...]:
    root = regular_path(root, directory=True)
    assets: list[Asset] = []
    seen: set[str] = set()
    entries = 0
    total = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal entries, total
        if depth > MAX_DEPTH:
            fail("INSPECTION_LIMIT", f"Folder depth exceeds {MAX_DEPTH}")
        with os.scandir(directory) as children:
            for entry in children:
                entries += 1
                if entries > MAX_FILES:
                    fail("INSPECTION_LIMIT", f"Folder exceeds {MAX_FILES} entries")
                path = Path(entry.path)
                name = relative_name(path.relative_to(root).as_posix())
                if name.lower() in seen:
                    fail("DUPLICATE_ASSET", f"Case-insensitive asset collision: {name}")
                seen.add(name.lower())
                # Windows scandir metadata can omit the file ID; compare two lstat results.
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    fail("SYMLINK_ASSET", f"Symlink/reparse point is not allowed: {name}")
                if stat.S_ISDIR(info.st_mode):
                    if entry.name in {".git", ".cache"}:
                        fail("INVALID_ASSET", f"Consumer skips {name}; use a clean prebuilt folder")
                    walk(path, depth + 1)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    if total > MAX_TOTAL:
                        fail("INSPECTION_LIMIT", f"Folder exceeds {MAX_TOTAL} bytes")
                    digest = sha256_file(path)
                    after = path.lstat()
                    if (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        fail("ASSET_CHANGED", f"Asset changed while hashing: {name}")
                    assets.append(Asset(name, info.st_size, digest))
                else:
                    fail("INVALID_ASSET", f"Nonregular asset: {name}")
    walk(root, 0)
    return tuple(sorted(assets, key=lambda asset: asset.path))


def content_fingerprint(assets: tuple[Asset, ...]) -> str:
    data = json.dumps([asdict(asset) for asset in assets], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("DUPLICATE_JSON_KEY", f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    fail("INVALID_JSON", f"Nonfinite JSON value: {value}")


def read_object(path: Path, limit: int = MAX_JSON) -> dict[str, Any]:
    regular_path(path)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        fail("INVALID_JSON", f"{path.name} must be nonempty and at most {limit} bytes")
    try:
        # Bound nesting before invoking the recursive JSON decoder.
        depth = 0
        quoted = escaped = False
        for char in data.decode("utf-8"):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                depth += 1
                if depth > MAX_DEPTH:
                    fail("INSPECTION_LIMIT", f"{path.name} nesting exceeds {MAX_DEPTH}")
            elif char in "}]":
                depth -= 1
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        fail("INVALID_JSON", f"Cannot parse {path.name}: {exc}")
    if not isinstance(value, dict):
        fail("INVALID_JSON", f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("INVALID_CONFIG", f"{label} must be an object")
    return cast(dict[str, Any], value)


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value <= 2147483647:
        fail("INVALID_CONFIG", f"{label} must be a positive int32")
    return int(value)


def _require_asset(root: Path, assets: dict[str, Asset], name: Any) -> str:
    name = relative_name(name)
    if name not in assets or assets[name].size == 0:
        fail("MISSING_ASSET", f"Missing, empty or incorrectly cased referenced asset: {name}")
    regular_path(root.joinpath(*PurePosixPath(name).parts))
    return cast(str, name)


def _providers(session: Any, label: str) -> tuple[str, dict[str, Any]]:
    session = _object(session, label)
    fallback = session.get("session.disable_cpu_ep_fallback", "")
    if fallback not in ("", "0", "1"):
        fail("INVALID_CONFIG", f"{label}: invalid session.disable_cpu_ep_fallback")
    groups = session.get("provider_options", [])
    if not isinstance(groups, list):
        fail("INVALID_CONFIG", f"{label}.provider_options must be an array")
    providers: dict[str, dict[str, Any]] = {}
    for group in groups:
        for name, options in _object(group, label).items():
            if name.lower() not in {"cpu", "qnn", "dml"} or name.lower() in providers:
                fail("INVALID_CONFIG", f"{label}: duplicate/unsupported provider {name!r}")
            options = _object(options, label)
            for key, value in options.items():
                if key == "device_filtering_options":
                    device = _object(value, key)
                    if set(device) - {"hardware_device_type", "hardware_device_id", "hardware_vendor_id"}:
                        fail("INVALID_CONFIG", f"{label}: unknown device filtering option")
                    device_type = device.get("hardware_device_type", "")
                    if not isinstance(device_type, str) or device_type.lower() not in {"", "cpu", "gpu", "npu"}:
                        fail("INVALID_CONFIG", f"{label}: invalid hardware_device_type")
                    for field_name in ("hardware_device_id", "hardware_vendor_id"):
                        number = device.get(field_name)
                        if number is not None and (type(number) is not int or not 0 <= number <= 0xFFFFFFFF):
                            fail("INVALID_CONFIG", f"{label}: invalid {field_name}")
                elif not isinstance(value, str):
                    fail("INVALID_CONFIG", f"{label}: provider option {key} must be a string")
            providers[name.lower()] = options
    if len(providers) > 1:
        fail("INVALID_CONFIG", f"{label}: multiple providers are outside this consumer route")
    return next(iter(providers.items()), ("cpu", {}))


def _config(root: Path, assets: dict[str, Asset]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_asset(root, assets, "genai_config.json")
    cfg = read_object(root / "genai_config.json")
    sanity = validate_genai_config(root)
    if sanity.errors:
        fail("INVALID_CONFIG", "; ".join(sanity.errors))
    model = _object(cfg.get("model"), "model")
    context = _positive(model.get("context_length"), "model.context_length")
    vocab = _positive(model.get("vocab_size"), "model.vocab_size")
    if not isinstance(model.get("type"), str) or not model["type"]:
        fail("INVALID_CONFIG", "model.type is required")
    eos = model.get("eos_token_id", model.get("pad_token_id", 0))
    tokens = eos if isinstance(eos, list) else [eos]
    if not tokens:
        fail("INVALID_CONFIG", "model.eos_token_id must not be empty")
    for token in [*tokens, model.get("pad_token_id", 0)]:
        if type(token) is not int or not 0 <= token < vocab:
            fail("INVALID_CONFIG", "Special token IDs must be within model.vocab_size")
    search = _object(cfg.get("search", {}), "search")
    if "max_length" in search and _positive(search["max_length"], "search.max_length") > context:
        fail("INVALID_CONFIG", "search.max_length exceeds model.context_length")
    decoder = _object(model.get("decoder"), "model.decoder")
    top_provider = _providers(decoder.get("session_options", {}), "decoder.session_options")
    window = decoder.get("sliding_window")
    if window is not None:
        window = _object(window, "sliding_window")
        if _positive(window.get("window_size"), "window_size") > context:
            fail("INVALID_CONFIG", "sliding_window exceeds context length")
        if window.get("alignment", "") not in {"", "left", "right"}:
            fail("INVALID_CONFIG", "sliding_window.alignment must be left or right")
        if "slide_inputs" in window and type(window["slide_inputs"]) is not bool:
            fail("INVALID_CONFIG", "slide_inputs must be boolean")
    pipeline = decoder.get("pipeline", [])
    if not isinstance(pipeline, list) or len(pipeline) > 64:
        fail("INVALID_CONFIG", "decoder.pipeline must be an array of at most 64 stage groups")
    stages: list[dict[str, Any]] = []
    seen: set[str] = set()
    if pipeline:
        if decoder.get("filename") or model["type"] != "decoder-pipeline":
            fail("INVALID_CONFIG", "Pipeline requires model.type=decoder-pipeline and no decoder.filename")
        for group in pipeline:
            group = _object(group, "pipeline group")
            if not group:
                fail("INVALID_CONFIG", "Empty pipeline group")
            for name, stage in group.items():
                if not name or name in seen or len(seen) >= 64:
                    fail("DUPLICATE_STAGE", f"Invalid/duplicate pipeline stage: {name!r}")
                seen.add(name)
                stage = _object(stage, name)
                provider, options = (_providers(stage["session_options"], name)
                                     if "session_options" in stage else top_provider)
                stages.append({"name": name, "filename": stage.get("filename"),
                               "provider": provider, "options": options})
    else:
        if model["type"] == "decoder-pipeline":
            fail("INVALID_CONFIG", "decoder-pipeline requires graph stages")
        stages.append({"name": "decoder", "filename": decoder.get("filename"),
                       "provider": top_provider[0], "options": top_provider[1]})
    references: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "filename":
                    if not isinstance(child, str):
                        fail("INVALID_REFERENCE", "Config filename references must be strings")
                    if child:
                        references.append(_require_asset(root, assets, child))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(model)
    if len({name.lower() for name in references}) != len(references):
        fail("DUPLICATE_STAGE", "Config references the same graph more than once")
    for stage in stages:
        _require_asset(root, assets, stage["filename"])
        if not stage["filename"].endswith(".onnx"):
            fail("INVALID_CONFIG", "Graph stages must reference .onnx files")
    # This bounded route is decoder-only, including CPU embedding/head pipeline stages.
    if set(references) != {stage["filename"] for stage in stages}:
        fail("UNSUPPORTED_CONFIG", "Only decoder/pipeline graph references are supported by this route")
    return cfg, stages


def _graphs(root: Path, assets: dict[str, Asset], stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import onnx
        from google.protobuf.message import DecodeError
    except (ImportError, OSError) as exc:
        fail("MISSING_ONNX", f"Structural graph validation requires optional onnx: {exc}. "
             "No graph validation or native execution was performed; supply an ONNX-capable environment.")
    results: list[dict[str, Any]] = []
    for stage in stages:
        name = stage["filename"]
        path = root.joinpath(*PurePosixPath(name).parts)
        if assets[name].size > MAX_GRAPH:
            fail("INSPECTION_LIMIT", f"{name} exceeds the {MAX_GRAPH}-byte graph metadata limit")
        try:
            model = onnx.load(str(path), load_external_data=False)
        except (DecodeError, OSError, ValueError) as exc:
            fail("INVALID_GRAPH", f"Cannot parse ONNX {name}: {exc}")
        if not model.HasField("graph") or not model.graph.node or not model.ir_version:
            fail("INVALID_GRAPH", f"{name}: expected a nonempty ONNX model graph and IR version")
        external: set[str] = set()
        contexts = caches = 0
        visited = 0

        def tensor(value: Any) -> None:
            if value.data_location != onnx.TensorProto.EXTERNAL:
                # QNN wrappers can retain stale metadata after embedding tensors.
                # data_location is authoritative; the checker validates embedded data.
                return
            metadata: dict[str, str] = {}
            for item in value.external_data:
                if item.key in metadata:
                    fail("INVALID_GRAPH", f"{name}: duplicate external data key {item.key}")
                metadata[item.key] = item.value
            location = relative_name(metadata.get("location"))
            joined = (PurePosixPath(name).parent / location).as_posix()
            _require_asset(root, assets, joined)
            external.add(joined)
            for key in ("offset", "length"):
                if key in metadata and not re.fullmatch(r"[0-9]{1,20}", metadata[key]):
                    fail("INVALID_GRAPH", f"{name}: invalid external data {key}")
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", str(assets[joined].size - offset)))
            if length <= 0 or offset < 0 or offset + length > assets[joined].size:
                fail("INVALID_GRAPH", f"{name}: external data range exceeds {joined}")

        def graph(value: Any, depth: int = 0) -> None:
            nonlocal contexts, caches, visited
            visited += len(value.node) + len(value.initializer)
            if depth > MAX_DEPTH or visited > 100000:
                fail("INSPECTION_LIMIT", f"{name}: graph nesting/node count exceeds bounds")
            for item in value.initializer:
                tensor(item)
            for item in value.sparse_initializer:
                tensor(item.values)
                tensor(item.indices)
            for node in value.node:
                attributes: dict[str, Any] = {}
                for attr in node.attribute:
                    if attr.name in attributes:
                        fail("INVALID_GRAPH", f"{name}: duplicate ONNX attribute {attr.name!r}")
                    attributes[attr.name] = attr
                    if attr.HasField("t"):
                        tensor(attr.t)
                    for item in attr.tensors:
                        tensor(item)
                    if attr.HasField("sparse_tensor"):
                        tensor(attr.sparse_tensor.values)
                        tensor(attr.sparse_tensor.indices)
                    for item in attr.sparse_tensors:
                        tensor(item.values)
                        tensor(item.indices)
                    if attr.HasField("g"):
                        graph(attr.g, depth + 1)
                    for item in attr.graphs:
                        graph(item, depth + 1)
                if node.op_type != "EPContext" or node.domain != "com.microsoft":
                    continue
                contexts += 1
                for key in ("source", "ep_cache_context"):
                    if key in attributes and attributes[key].type != onnx.AttributeProto.STRING:
                        fail("INVALID_GRAPH", f"{name}: EPContext {key} must be a string")
                source = attributes.get("source")
                if source is None or source.s.lower() not in {b"qnn", b"qnnexecutionprovider"}:
                    fail("INVALID_GRAPH", f"{name}: EPContext source is not QNN")
                for key in ("embed_mode", "main_context"):
                    if key in attributes and (attributes[key].type != onnx.AttributeProto.INT
                                              or attributes[key].i not in (0, 1)):
                        fail("INVALID_GRAPH", f"{name}: invalid EPContext {key}")
                cache = attributes.get("ep_cache_context")
                main = attributes.get("main_context")
                if cache is None or not cache.s:
                    if main is not None and main.i == 1:
                        fail("MISSING_ASSET", f"{name}: main EPContext has no cache")
                    continue
                caches += 1
                embed = attributes.get("embed_mode")
                if embed is not None and embed.i == 0:
                    try:
                        cache_name = cache.s.decode("utf-8")
                    except UnicodeError as exc:
                        fail("INVALID_REFERENCE", f"{name}: invalid context path: {exc}")
                    # The fork resolves EPContext cache paths against the model root.
                    external.add(_require_asset(root, assets, cache_name))
        graph(model.graph)
        if model.functions or model.training_info:
            fail("UNSUPPORTED_GRAPH", f"{name}: local functions/training graphs are outside this route")
        if contexts and (not caches or stage["provider"] != "qnn"):
            fail("INVALID_GRAPH", f"{name}: EPContext requires a QNN stage and nonempty context")
        if stage["provider"] == "qnn" and not contexts:
            fail("QNN_CONTEXT_REQUIRED", f"{name}: this prebuilt QNN route requires EPContext, not raw QNN compilation")
        try:
            # Path-based checking supplies the correct external-data directory without loading weights.
            onnx.checker.check_model(str(path), full_check=False)
        except (onnx.checker.ValidationError, OSError, ValueError) as exc:
            fail("INVALID_GRAPH", f"ONNX structural checker rejected {name}: {exc}")
        results.append({"filename": name, "provider": stage["provider"], "ep_context_nodes": contexts,
                        "external_assets": sorted(external)})
    return results


def _text(value: Any, limit: int) -> bool:
    return (isinstance(value, str) and 0 < len(value.encode("utf-8")) <= limit
            and value.strip() == value and not any(unicodedata.category(c) == "Cc" for c in value))


def _declaration(root: Path, cfg: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    p = read_object(root / "ollama_npu.json", 64 << 10)
    required = {"schema_version", "platform", "provider", "genai_version", "ort_version",
                "processor_names", "context_length", "precision", "capabilities", "stages"}
    if set(p) - required - {"source"} or required - set(p):
        fail("INVALID_DECLARATION", "ollama_npu.json has unknown or missing schema-1 fields")
    if type(p["schema_version"]) is not int or p["schema_version"] != 1:
        fail("INVALID_DECLARATION", "Only ollama_npu.json schema_version=1 is supported")
    if p["platform"] != "windows/arm64" or p["provider"] != "qnn":
        fail("INVALID_DECLARATION", "Schema 1 requires platform=windows/arm64 and provider=qnn")
    for name in ("genai_version", "ort_version"):
        if not isinstance(p[name], str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", p[name]):
            fail("INVALID_DECLARATION", f"{name} must be exact major.minor.patch")
    if not _text(p["precision"], 64) or p["capabilities"] != ["completion"]:
        fail("INVALID_DECLARATION", "Declared precision and capabilities=[completion] are required")
    _positive(p["context_length"], "declared context_length")
    if p["context_length"] != cfg["model"]["context_length"]:
        fail("CONTRADICTORY_DECLARATION", "Declared context_length differs from genai_config.json")
    processors = p["processor_names"]
    if (not isinstance(processors, list) or not 1 <= len(processors) <= 32
            or any(not _text(name, 256) for name in processors)
            or len({name.lower() for name in processors}) != len(processors)):
        fail("INVALID_DECLARATION", "processor_names requires 1..32 unique exact processor names")
    declared_stages = p["stages"]
    if not isinstance(declared_stages, list) or not 1 <= len(declared_stages) <= 64:
        fail("INVALID_DECLARATION", "stages requires 1..64 graph stages")
    actual = {stage["filename"]: stage["provider"] for stage in stages}
    declared: dict[str, str] = {}
    for stage in declared_stages:
        stage = _object(stage, "declared stage")
        if set(stage) != {"filename", "provider"}:
            fail("INVALID_DECLARATION", "Stage requires exactly filename and provider")
        name = relative_name(stage["filename"])
        if (not name.endswith(".onnx") or name.lower() in {key.lower() for key in declared}
                or stage["provider"] not in ("cpu", "qnn")):
            fail("INVALID_DECLARATION", "Invalid/duplicate declared graph stage")
        declared[name] = stage["provider"]
    if declared != actual or "qnn" not in declared.values():
        fail("CONTRADICTORY_DECLARATION", "Declaration must describe every config graph/provider exactly once with QNN")
    for stage in stages:
        if stage["provider"] == "qnn":
            for key, expected in {"backend_type": "htp", "backend_path": "QnnHtp.dll"}.items():
                if stage["options"].get(key, expected) != expected:
                    fail("CONTRADICTORY_DECLARATION", f"Schema-1 QNN requires {key}={expected}")
    source = p.get("source")
    if source is not None:
        source = _object(source, "source")
        if set(source) != {"repository", "revision"} or not _text(source["repository"], 1024):
            fail("INVALID_DECLARATION", "source requires repository and revision only")
        url = urlsplit(source["repository"])
        if (url.scheme != "https" or not url.netloc or url.username is not None or url.password is not None
                or url.path in ("", "/") or url.query or url.fragment or "?" in source["repository"]
                or not isinstance(source["revision"], str)
                or not re.fullmatch(r"[a-f0-9]{40}([a-f0-9]{24})?", source["revision"])):
            fail("INVALID_DECLARATION", "source requires credential-free HTTPS repo and full lowercase Git revision")
    return p


@dataclass
class Inspection:
    input: str
    assets: tuple[Asset, ...] = ()
    content_fingerprint: str | None = None
    structural: str = "failed"
    declaration_status: str = "not-checked"
    declaration: dict[str, Any] | None = None
    target: dict[str, Any] = field(default_factory=lambda: {"status": "not-checked"})
    graphs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    context_length: int | None = None

    @property
    def importable(self) -> bool:
        return (self.structural == "passed" and self.declaration_status in {"absent", "consistent"}
                and self.target["status"] != "mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "input": self.input, "assets": [asdict(asset) for asset in self.assets],
                "content_fingerprint": self.content_fingerprint, "importable": self.importable,
                "context_length": self.context_length, "graphs": self.graphs, "errors": self.errors,
                "levels": {"structural": {"status": self.structural},
                           "declaration": {"status": self.declaration_status, "profile": self.declaration},
                           "target": self.target,
                           "native_evidence": {"status": "not-run", "certified": False,
                                               "detail": "No runtime load, generation, profiling or supplier certification."}}}


def inspect_prebuilt(root: Path, *, processor_name: str | None = None) -> Inspection:
    report = Inspection(input=str(root.absolute()))

    def error(level: str, exc: TemperumError) -> None:
        report.errors.append({"level": level, "code": exc.reason_code, "message": str(exc)})

    try:
        root = regular_path(root, directory=True)
        report.input = str(root)
        report.assets = inventory(root)
        assets = {asset.path: asset for asset in report.assets}
        report.content_fingerprint = content_fingerprint(report.assets)
        for asset in report.assets:
            if asset.path.lower() == "ollama_npu.json" and asset.path != "ollama_npu.json":
                fail("INVALID_DECLARATION", "Profile must be named exactly ollama_npu.json")
            if Path(asset.path).suffix.lower() in {".gguf", ".safetensors", ".dll", ".exe", ".lib", ".pdb"}:
                fail("INVALID_ASSET", f"Mixed format/native executable in prebuilt folder: {asset.path}")
            if asset.path.endswith(".onnx") and asset.size == 0:
                fail("INVALID_GRAPH", f"Empty ONNX graph: {asset.path}")
        cfg, stages = _config(root, assets)
        report.context_length = cfg["model"]["context_length"]
        tokenizers = [name for name in ("tokenizer.json", "tokenizer.model") if name in assets]
        if not tokenizers:
            fail("MISSING_TOKENIZER", "Requires root tokenizer.json or tokenizer.model")
        for name in tokenizers:
            _require_asset(root, assets, name)
            if name.endswith(".json"):
                read_object(root / name, 64 << 20)
        for name in ("tokenizer_config.json", "config.json", "generation_config.json", "special_tokens_map.json"):
            if name in assets:
                read_object(root / name)
    except TemperumError as exc:
        error("structural", exc)
        return report
    except (OSError, ValueError) as exc:
        error("structural", TemperumError("prebuilt", "READ_FAILED", str(exc)))
        return report
    try:
        # Check all ONNX assets, not just referenced graph names.
        extra = [asset.path for asset in report.assets
                 if asset.path.endswith(".onnx") and asset.path not in {s["filename"] for s in stages}]
        if extra:
            fail("UNREFERENCED_GRAPH", f"Unreferenced ONNX assets: {extra}")
        report.graphs = _graphs(root, assets, stages)
        report.structural = "passed"
    except TemperumError as exc:
        error("structural", exc)
    except OSError as exc:
        error("structural", TemperumError("prebuilt", "READ_FAILED", str(exc)))
    report.declaration_status = "absent"
    if "ollama_npu.json" in assets:
        report.declaration_status = "invalid"
        try:
            report.declaration = _declaration(root, cfg, stages)
            report.declaration_status = "consistent"
        except TemperumError as exc:
            error("declaration", exc)
        except (OSError, ValueError) as exc:
            error("declaration", TemperumError("prebuilt", "INVALID_DECLARATION", str(exc)))
    if report.declaration is not None:
        mismatches = [key for key, value in CONSUMER.items() if report.declaration[key] != value]
        if processor_name is not None and processor_name.strip().lower() not in {
                name.lower() for name in report.declaration["processor_names"]}:
            mismatches.append("processor_name")
        report.target = {"status": "mismatch" if mismatches else (
            "declared-match" if processor_name is not None else "runtime-match-processor-unchecked"),
            "consumer": CONSUMER, "processor_name": processor_name, "mismatches": mismatches,
            "detail": "Declaration comparison only; host and native execution are not probed."}
    elif processor_name is not None:
        report.target = {"status": "unverified", "processor_name": processor_name,
                         "detail": "No valid target declaration; native/target compatibility is unverified."}
    return report
