"""Targets (the hardware a bundle is compiled for) and runtimes (what executes it), as data.

Adding a SoC, NPU vendor or runtime version should mean adding a profile here, not a new
code path. Values were validated in the M0 spike (spike/m0/RESULTS.md) unless marked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from npu_model.core.errors import NpuModelError


@dataclass(frozen=True)
class TargetProfile:
    id: str
    description: str
    vendor: str
    provider: str  # execution-provider family the runtime uses for NPU stages
    soc_model: str  # QNN EP `soc_model` for ahead-of-time context compilation
    htp_arch: str
    processor_patterns: tuple[str, ...]  # regexes matched against the CPU name string
    verified: bool  # a conversion has been run and checked on this hardware

    def matches(self, processor_name: str | None) -> bool:
        return bool(processor_name) and any(
            re.search(p, processor_name or "", re.IGNORECASE) for p in self.processor_patterns)


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    description: str
    # Versions recorded in ollama_npu.json; the fork checks them before loading.
    genai_version: str
    ort_version: str
    qnn_version: str
    # Packages for the compile role: the QNN EP here must match the runtime's QNN SDK,
    # because contexts compiled by a newer QNN generally do not load on an older runtime.
    compile_requirements: tuple[str, ...]
    # Packages for the verify role (a Python copy of the consumer runtime).
    verify_requirements: tuple[str, ...]
    verify_requirements_nodeps: tuple[str, ...]
    # GenAI whose ModelBuilder writes configs this runtime can read.
    builder_genai: str


TARGETS: dict[str, TargetProfile] = {
    "qnn-x-elite": TargetProfile(
        id="qnn-x-elite",
        description="Snapdragon X Elite / X Plus, Hexagon NPU (HTP v73)",
        vendor="qualcomm",
        provider="qnn",
        soc_model="60",
        htp_arch="v73",
        processor_patterns=(r"\bX1E[-\d]", r"\bX1P[-\d]"),
        verified=True,  # X1E78100 only; X Plus SKUs are expected but unverified
    ),
    "qnn-x2-elite": TargetProfile(
        id="qnn-x2-elite",
        description="Snapdragon X2 Elite, Hexagon NPU (HTP v81) — compile-only, unverified",
        vendor="qualcomm",
        provider="qnn",
        soc_model="87",
        htp_arch="v81",
        processor_patterns=(r"\bX2E[-\d]", r"\bX2P[-\d]"),
        verified=False,
    ),
}

RUNTIMES: dict[str, RuntimeProfile] = {
    "fork-qnn-1.24.1": RuntimeProfile(
        id="fork-qnn-1.24.1",
        description="Ollama NPU fork pinned runtime: ORT GenAI 0.12.2 / ORT QNN 1.24.1 (QNN 2.42.0)",
        genai_version="0.12.2",
        ort_version="1.24.1",
        qnn_version="2.42.0",
        compile_requirements=("olive-ai==0.13.0", "onnxruntime-qnn==1.24.1", "requests"),
        verify_requirements=("onnxruntime-qnn==1.24.1", "numpy"),
        verify_requirements_nodeps=("onnxruntime-genai==0.12.2",),
        builder_genai="0.12.2",
    ),
}

DEFAULT_RUNTIME = "fork-qnn-1.24.1"


def get_target(target_id: str) -> TargetProfile:
    try:
        return TARGETS[target_id]
    except KeyError:
        raise NpuModelError("target", "UNKNOWN_TARGET", f"Unknown target profile: {target_id}",
                            hint=f"Known targets: {', '.join(sorted(TARGETS))}") from None


def get_runtime(runtime_id: str) -> RuntimeProfile:
    try:
        return RUNTIMES[runtime_id]
    except KeyError:
        raise NpuModelError("runtime", "UNKNOWN_RUNTIME", f"Unknown runtime profile: {runtime_id}",
                            hint=f"Known runtimes: {', '.join(sorted(RUNTIMES))}") from None
