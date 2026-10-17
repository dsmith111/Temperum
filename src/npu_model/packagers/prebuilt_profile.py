"""Explicit, offline derivation of one content-pinned engineering model profile."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, NoReturn

from npu_model.core.errors import NpuModelError
from npu_model.inspect.prebuilt import (
    Asset, content_fingerprint, inspect_prebuilt, inventory, read_object, regular_path, relative_name,
)

PROFILE = "phi3-mini-4k-qnn"
REPOSITORY = "https://huggingface.co/llmware/phi-3-mini-4k-instruct-onnx-qnn"
REVISION = "6fa72cea4103de0bc69792a6655be538a8c4d7e0"
SOURCE_FINGERPRINT = "6fa2fa387024705b616ac12f9468e6671502e24468420eaa3095d3c617cac653"
PROCESSOR = "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU"
MODIFIED_LOG_ID = "npu-model modified genai_config.json: file-mapping disabled; see NOTICE.txt"
STAGES = (("embeddings.onnx", "cpu"), ("context_ctx.onnx", "qnn"),
          ("iterator_ctx.onnx", "qnn"), ("lm_head.onnx", "cpu"))
TEMPLATE = "{{ range .Messages }}<|{{ .Role }}|>\n{{ .Content }}<|end|>\n{{ end }}<|assistant|>\n"


def _fail(code: str, message: str) -> NoReturn:
    raise NpuModelError("profile", code, message)


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _asset(name: str, data: bytes) -> Asset:
    return Asset(name, len(data), hashlib.sha256(data).hexdigest())


def _destination(path: Path, source: Path) -> Path:
    if ".." in path.parts:
        _fail("INVALID_DESTINATION", "Destination must not contain traversal.")
    path = Path(os.path.abspath(path.expanduser()))
    relative_name(path.name)
    regular_path(path.parent, directory=True)
    if path == source or path.is_relative_to(source):
        _fail("INVALID_DESTINATION", "Derived output must be outside the immutable source.")
    if os.path.lexists(path):
        _fail("DESTINATION_EXISTS", "Profile output already exists; nothing will be overwritten.")
    return path


@dataclass(frozen=True)
class ProfilePlan:
    source: str
    destination: str
    assets: tuple[Asset, ...]
    generated: tuple[tuple[str, bytes], ...]

    def identity(self) -> dict[str, Any]:
        return {"profile": PROFILE, "schema_version": 1, "source": self.source, "destination": self.destination,
                "source_repository": REPOSITORY, "source_revision": REVISION,
                "source_fingerprint": content_fingerprint(self.assets),
                "generated": [asdict(_asset(name, data)) for name, data in self.generated],
                "platform": "windows/arm64", "processor_name": PROCESSOR, "overwrite": False}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json(self.identity())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"action": "plan-prepare-prebuilt", "executed": False, "fingerprint": self.fingerprint,
                "plan": self.identity(), "native_evidence": "not-run",
                "limits": "Engineering profile: 512-token default; 4096 is compiled capacity, not quality approval. "
                          "No agent/tool/structured-output certification or reproduced producer recipe.",
                "next": "Review notices and plan; repeat with --execute --expect-fingerprint HASH --accept-license."}


def plan_profile(source: Path, destination: Path, *, profile: str) -> ProfilePlan:
    """No network, native load, remote Python import or writes."""
    if profile != PROFILE:
        _fail("UNKNOWN_PROFILE", f"Only the explicitly selected {PROFILE!r} engineering profile is available.")
    try:
        source = regular_path(source, directory=True)
        destination = _destination(destination, source)
        report = inspect_prebuilt(source)
        if not report.importable:
            _fail("INSPECTION_FAILED", "Source must pass inspect-prebuilt before profile preparation.")
        if report.content_fingerprint != SOURCE_FINGERPRINT:
            _fail("SOURCE_MISMATCH", "Source does not match the complete 23-file pinned upstream inventory. "
                  "Acquire the exact revision; modified, incomplete or extra files are not accepted.")
        if [(graph["filename"], graph["provider"]) for graph in report.graphs] != list(STAGES):
            _fail("SOURCE_MISMATCH", "Source graph/provider stages differ from the reviewed profile.")
        config = read_object(source / "genai_config.json")
        modified: set[str] = set()
        for group in config["model"]["decoder"]["pipeline"]:
            for stage in group.values():
                for provider in stage.get("session_options", {}).get("provider_options", []):
                    if "qnn" in provider:
                        provider["qnn"]["disable_file_mapped_weights"] = "1"
                        modified.add(stage["filename"])
        if modified != {name for name, provider in STAGES if provider == "qnn"}:
            _fail("SOURCE_MISMATCH", "Expected exactly two explicit QNN sessions.")
        # A supported log_id carries the changed-file notice inside valid GenAI JSON.
        config["model"]["decoder"]["session_options"]["log_id"] = MODIFIED_LOG_ID
        declaration = {"schema_version": 1, "platform": "windows/arm64", "provider": "qnn",
                       "genai_version": "0.12.2", "ort_version": "1.24.1", "processor_names": [PROCESSOR],
                       "context_length": 4096, "precision": "int4 (upstream card declaration)",
                       "capabilities": ["completion"],
                       "stages": [{"filename": name, "provider": provider} for name, provider in STAGES],
                       "source": {"repository": REPOSITORY, "revision": REVISION}}
        data = files("npu_model.packagers").joinpath("profile_data")
        apache = data.joinpath("APACHE-2.0.txt").read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        microsoft = data.joinpath("MICROSOFT-MIT.txt").read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        notice = (
            f"Derived engineering profile: {PROFILE}, revision 1\n"
            f"Source: {REPOSITORY}\nRevision: {REVISION}\n"
            f"Complete source content fingerprint: {SOURCE_FINGERPRINT}\n"
            "Upstream model card declares Apache-2.0, int4, QAIRT 2.36 / ORT 1.22.2 / GenAI 0.9.\n"
            "README.md and all source assets/notices are retained; Python files are inert text, never executed.\n"
            "Source headers: Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.\n"
            "No standalone upstream LICENSE or NOTICE was present in this pinned repository.\n"
            "licenses/APACHE-2.0.txt supplies the declared license text.\n"
            "licenses/MICROSOFT-MIT.txt also retains the Phi-3 Mini 4K base-family license, obtained from\n"
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/"
            "f39ac1d28e925b323eae81227eaba4464caced4e/LICENSE\n"
            "That reference is not a claim that this compiled prebuilt used those exact base weights.\n"
            "\nLOCAL CHANGES by npu-model profile revision 1:\n"
            "genai_config.json: disable_file_mapped_weights=\"1\" in both explicit QNN pipeline sessions;\n"
            "decoder session log_id now carries a prominent changed-file notice. JSON is reserialized as LF UTF-8.\n"
            "The unchanged original is retained at source/genai_config.json; other upstream files are unchanged.\n"
            "Upstream hash_record_sha256.json describes original producer files, NOT this derivative.\n"
            "Added this notice, license copies, source_receipt.json, ollama_npu.json and LF Modelfile.\n"
            "Modelfile attaches the reviewed Phi role template, num_ctx=512 and num_predict=128.\n"
            "Compatibility declaration is by this derivative packager, not llmware/Microsoft/Qualcomm certification.\n"
            f"Target: {PROCESSOR}; Windows ARM64, GenAI QNN 0.12.2 / ORT QNN 1.24.1.\n"
            "CPU embeddings/head are intentional. Capacity 4096 is NOT useful-long-context qualification.\n"
            "No tools/agent/structured-output claim, producer rebuild, complete rights-chain or native proof here.\n"
            "No runtime DLLs are included; their separate terms/notices still apply. No publication was performed.\n"
        ).encode("utf-8")
        license_text = b"\n\n".join((notice, apache, microsoft)).decode("utf-8")
        modelfile = (f'FROM .\nTEMPLATE """{TEMPLATE}"""\nPARAMETER num_ctx 512\nPARAMETER num_predict 128\n'
                     f'LICENSE """{license_text}"""\n').encode("utf-8")
        receipt = {"schema_version": 1, "profile": PROFILE, "profile_revision": 1,
                   "source": {"repository": REPOSITORY, "revision": REVISION, "content_fingerprint": SOURCE_FINGERPRINT,
                              "assets": [asdict(asset) for asset in report.assets]},
                   "native_evidence": {"status": "not-run", "certified": False}}
        # Pipeline groups execute in member order; canonical key sorting would reorder stages.
        config_bytes = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode("utf-8")
        generated = {"genai_config.json": config_bytes, "source/genai_config.json": (source / "genai_config.json").read_bytes(),
                     "ollama_npu.json": _json(declaration), "NOTICE.txt": notice,
                     "licenses/APACHE-2.0.txt": apache, "licenses/MICROSOFT-MIT.txt": microsoft,
                     "source_receipt.json": _json(receipt), "Modelfile": modelfile}
        return ProfilePlan(str(source), str(destination), report.assets, tuple(sorted(generated.items())))
    except OSError as exc:
        raise NpuModelError("profile", "LOCAL_IO_FAILED", "Cannot read/plan the local profile.", cause=exc) from exc


def execute_profile(plan: ProfilePlan, *, expected_fingerprint: str | None, accept_license: bool) -> dict[str, Any]:
    if expected_fingerprint != plan.fingerprint:
        _fail("PLAN_CHANGED", "Execution requires the exact reviewed profile plan fingerprint.")
    if not accept_license:
        _fail("RIGHTS_CONSENT_REQUIRED", "Review licenses/notices and confirm use rights with --accept-license.")
    if os.name != "nt":
        _fail("ATOMIC_PROMOTION_UNAVAILABLE", "This profile uses Windows same-volume no-replace directory rename.")
    stage: Path | None = None
    try:
        fresh = plan_profile(Path(plan.source), Path(plan.destination), profile=PROFILE)
        if fresh.fingerprint != plan.fingerprint:
            _fail("PLAN_CHANGED", "Source, profile or output plan changed; no output was written.")
        source, destination = Path(plan.source), Path(plan.destination)
        candidate = destination.parent / ("npu-profile-" + uuid.uuid4().hex)
        candidate.mkdir()
        stage = candidate
        for asset in plan.assets:
            target = stage / asset.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular_path(source / asset.path), target)
        if inventory(stage) != plan.assets or inventory(source) != plan.assets:
            _fail("SOURCE_CHANGED", "Source changed during snapshotting; no completed profile was promoted.")
        for name, content in plan.generated:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        expected = {asset.path: asset for asset in plan.assets}
        expected.update({name: _asset(name, content) for name, content in plan.generated})
        report = inspect_prebuilt(stage, processor_name=PROCESSOR)
        if not report.importable or report.assets != tuple(sorted(expected.values(), key=lambda asset: asset.path)):
            _fail("PROFILE_INVALID", "Derived output failed structural, declaration or exact inventory validation.")
        _destination(destination, source)
        regular_path(stage, directory=True)
        stage.rename(destination)
        stage = None
        return {"action": "prepare-prebuilt", "executed": True, "destination": str(destination),
                "fingerprint": plan.fingerprint, "content_fingerprint": report.content_fingerprint,
                "assets": [asdict(asset) for asset in report.assets], "native_evidence": "not-run",
                "next": "Use import-ollama with this input; its included Modelfile needs no manual edits. "
                        "Run generation and native qualification separately; this is not a publication."}
    except OSError as exc:
        raise NpuModelError("profile", "LOCAL_IO_FAILED", "Profile preparation failed; existing output preserved.",
                            cause=exc) from exc
    finally:
        if stage is not None:
            active_error = sys.exc_info()[1]
            try:
                shutil.rmtree(stage)
            except OSError as exc:
                message = f"Cannot remove owned profile staging {stage}: {exc}. No ACL changes made."
                if isinstance(active_error, NpuModelError):
                    active_error.hint = message
                else:
                    _fail("STAGING_CLEANUP_FAILED", message)
