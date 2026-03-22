"""Hardware profile detection and optimization configuration.

This module detects the runtime hardware and provides optimization hints
that other parts of zipline can use to tune behavior. Profiles are designed
to be extensible — add new profiles by subclassing ``HardwareProfile``.

Usage::

    from zipline.utils.hardware_profile import get_profile

    profile = get_profile()
    print(profile.name)              # e.g. "apple_silicon"
    print(profile.blas_provider)     # e.g. "accelerate"
    print(profile.optimal_threads)   # e.g. 8
    print(profile.capabilities)      # set of capability strings

Profiles can also be forced via environment variable::

    export ZIPLINE_HARDWARE_PROFILE=apple_silicon
"""

import logging
import os
import platform
import struct
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability flags — used by optimization code to decide what to enable
# ---------------------------------------------------------------------------
CAP_NEON = "neon"                    # ARM NEON SIMD
CAP_AVX2 = "avx2"                   # x86 AVX2 SIMD
CAP_AVX512 = "avx512"               # x86 AVX-512
CAP_ACCELERATE = "apple_accelerate" # Apple Accelerate (vDSP/vecLib)
CAP_MKL = "mkl"                     # Intel MKL
CAP_OPENBLAS = "openblas"           # OpenBLAS
CAP_GPU_METAL = "gpu_metal"         # Apple Metal (GPU compute)
CAP_GPU_CUDA = "gpu_cuda"           # NVIDIA CUDA
CAP_GPU_ROCM = "gpu_rocm"           # AMD ROCm
CAP_MULTICORE = "multicore"         # >1 performance core available


@dataclass(frozen=True)
class HardwareProfile:
    """Description of the runtime hardware and its optimisation knobs."""

    name: str
    arch: str                     # "arm64", "x86_64", etc.
    system: str                   # "darwin", "linux", "windows"
    cpu_name: str                 # human-readable CPU identifier
    physical_cores: int
    logical_cores: int
    ram_gb: float
    blas_provider: str            # "accelerate", "mkl", "openblas", "unknown"
    capabilities: frozenset = field(default_factory=frozenset)

    # Tuning knobs — consumers can read these to configure behaviour
    optimal_threads: int = 1      # recommended thread count for compute
    sort_algorithm: str = "stable"  # numpy sort kind for ranking
    use_parallel_pipeline: bool = False  # enable threaded pipeline execution
    blosc_nthreads: int = 1       # blosc decompression threads

    def has(self, capability: str) -> bool:
        """Check if this profile has a specific capability."""
        return capability in self.capabilities

    def summary(self) -> str:
        """One-line human-readable summary."""
        caps = ", ".join(sorted(self.capabilities)) or "none"
        return (
            f"{self.name} | {self.arch} | {self.cpu_name} | "
            f"{self.physical_cores}P/{self.logical_cores}L cores | "
            f"{self.ram_gb:.1f} GB RAM | BLAS={self.blas_provider} | "
            f"caps=[{caps}]"
        )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _get_cpu_name() -> str:
    system = platform.system().lower()
    if system == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=5,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    if system == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _get_ram_gb() -> float:
    system = platform.system().lower()
    if system == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=5,
            ).strip()
            return int(out) / (1024 ** 3)
        except Exception:
            pass
    if system == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            pass
    return 0.0


def _detect_blas() -> str:
    """Detect the BLAS provider numpy is linked against."""
    try:
        config = np.__config__
        # NumPy 2.x
        if hasattr(config, "blas_ilp64_opt_info"):
            info = config.blas_ilp64_opt_info
        elif hasattr(config, "blas_opt_info"):
            info = config.blas_opt_info
        else:
            info = {}

        if isinstance(info, dict):
            libs = info.get("libraries", [])
            lib_str = " ".join(str(l) for l in libs).lower()
        else:
            lib_str = str(info).lower()

        if "accelerate" in lib_str or "veclib" in lib_str:
            return "accelerate"
        if "mkl" in lib_str:
            return "mkl"
        if "openblas" in lib_str:
            return "openblas"
        if "blis" in lib_str:
            return "blis"

        # NumPy 2.x show_config() returns a dict
        try:
            show = np.show_config(mode="dicts")
            if isinstance(show, dict):
                blas_info = show.get("Build Dependencies", {}).get("blas", {})
                blas_name = blas_info.get("name", "").lower()
                if "accelerate" in blas_name:
                    return "accelerate"
                if "mkl" in blas_name:
                    return "mkl"
                if "openblas" in blas_name:
                    return "openblas"
        except Exception:
            pass

    except Exception:
        pass
    return "unknown"


def _detect_x86_simd() -> set:
    """Detect x86 SIMD capabilities."""
    caps = set()
    system = platform.system().lower()

    if system == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read().lower()
            if "avx2" in content:
                caps.add(CAP_AVX2)
            if "avx512" in content:
                caps.add(CAP_AVX512)
        except Exception:
            pass
    elif system == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features"],
                text=True, timeout=5,
            ).lower()
            if "avx2" in out:
                caps.add(CAP_AVX2)
            if "avx512" in out:
                caps.add(CAP_AVX512)
        except Exception:
            pass
    return caps


def _detect_gpu() -> set:
    """Detect GPU compute capabilities."""
    caps = set()
    system = platform.system().lower()

    # Apple Metal
    if system == "darwin" and platform.machine() == "arm64":
        caps.add(CAP_GPU_METAL)

    # NVIDIA CUDA
    try:
        subprocess.check_output(
            ["nvidia-smi"], timeout=5, stderr=subprocess.DEVNULL,
        )
        caps.add(CAP_GPU_CUDA)
    except Exception:
        pass

    # AMD ROCm
    try:
        subprocess.check_output(
            ["rocm-smi"], timeout=5, stderr=subprocess.DEVNULL,
        )
        caps.add(CAP_GPU_ROCM)
    except Exception:
        pass

    return caps


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------

def _build_apple_silicon_profile(
    cpu_name: str,
    physical_cores: int,
    logical_cores: int,
    ram_gb: float,
    blas: str,
    gpu_caps: set,
) -> HardwareProfile:
    """Profile for Apple M1/M2/M3/M4 Macs."""
    caps = {CAP_NEON, CAP_MULTICORE}
    if blas == "accelerate":
        caps.add(CAP_ACCELERATE)
    caps |= gpu_caps

    # M-series has performance + efficiency cores.
    # Use perf core count for compute threads (typically half of logical).
    perf_cores = max(physical_cores // 2, physical_cores - 4)
    optimal_threads = max(perf_cores, 2)

    return HardwareProfile(
        name="apple_silicon",
        arch="arm64",
        system="darwin",
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        blas_provider=blas,
        capabilities=frozenset(caps),
        optimal_threads=optimal_threads,
        # Apple's sort implementations are well-optimized for ARM cache
        sort_algorithm="stable",
        use_parallel_pipeline=physical_cores >= 4,
        blosc_nthreads=min(optimal_threads, 4),
    )


def _build_linux_x86_profile(
    cpu_name: str,
    physical_cores: int,
    logical_cores: int,
    ram_gb: float,
    blas: str,
    simd_caps: set,
    gpu_caps: set,
) -> HardwareProfile:
    """Profile for Linux x86_64 machines (workstations/servers)."""
    caps = set(simd_caps) | gpu_caps
    if physical_cores > 1:
        caps.add(CAP_MULTICORE)
    if blas == "mkl":
        caps.add(CAP_MKL)
    elif blas == "openblas":
        caps.add(CAP_OPENBLAS)

    optimal_threads = max(physical_cores, 2)

    return HardwareProfile(
        name="linux_x86",
        arch="x86_64",
        system="linux",
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        blas_provider=blas,
        capabilities=frozenset(caps),
        optimal_threads=optimal_threads,
        sort_algorithm="stable",
        use_parallel_pipeline=physical_cores >= 4,
        blosc_nthreads=min(optimal_threads, 4),
    )


def _build_linux_arm_profile(
    cpu_name: str,
    physical_cores: int,
    logical_cores: int,
    ram_gb: float,
    blas: str,
    gpu_caps: set,
) -> HardwareProfile:
    """Profile for Linux aarch64 (Graviton, Ampere, etc.)."""
    caps = {CAP_NEON}
    if physical_cores > 1:
        caps.add(CAP_MULTICORE)
    caps |= gpu_caps

    return HardwareProfile(
        name="linux_arm",
        arch="aarch64",
        system="linux",
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        blas_provider=blas,
        capabilities=frozenset(caps),
        optimal_threads=max(physical_cores, 2),
        sort_algorithm="stable",
        use_parallel_pipeline=physical_cores >= 4,
        blosc_nthreads=min(physical_cores, 4),
    )


def _build_generic_profile(
    cpu_name: str,
    physical_cores: int,
    logical_cores: int,
    ram_gb: float,
    blas: str,
) -> HardwareProfile:
    """Fallback profile for unrecognized platforms."""
    caps = set()
    if physical_cores > 1:
        caps.add(CAP_MULTICORE)

    return HardwareProfile(
        name="generic",
        arch=platform.machine(),
        system=platform.system().lower(),
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        blas_provider=blas,
        capabilities=frozenset(caps),
        optimal_threads=max(physical_cores, 1),
        sort_algorithm="stable",
        use_parallel_pipeline=False,
        blosc_nthreads=1,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def detect_profile() -> HardwareProfile:
    """Auto-detect the current hardware and return an appropriate profile."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    cpu_name = _get_cpu_name()
    ram_gb = _get_ram_gb()
    blas = _detect_blas()
    gpu_caps = _detect_gpu()

    try:
        physical = os.cpu_count() or 1
        # On macOS, sysctl gives physical core count
        if system == "darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    text=True, timeout=5,
                ).strip()
                physical = int(out)
            except Exception:
                pass
        logical = os.cpu_count() or physical
    except Exception:
        physical = logical = 1

    if machine == "arm64" and system == "darwin":
        return _build_apple_silicon_profile(
            cpu_name, physical, logical, ram_gb, blas, gpu_caps,
        )

    if machine in ("x86_64", "amd64") and system == "linux":
        simd_caps = _detect_x86_simd()
        return _build_linux_x86_profile(
            cpu_name, physical, logical, ram_gb, blas, simd_caps, gpu_caps,
        )

    if machine == "aarch64" and system == "linux":
        return _build_linux_arm_profile(
            cpu_name, physical, logical, ram_gb, blas, gpu_caps,
        )

    return _build_generic_profile(cpu_name, physical, logical, ram_gb, blas)


_PROFILE_REGISTRY: dict[str, type] = {}
_forced_profile: Optional[HardwareProfile] = None


def get_profile() -> HardwareProfile:
    """Get the active hardware profile.

    Checks (in order):
    1. A profile forced via ``set_profile()``
    2. The ``ZIPLINE_HARDWARE_PROFILE`` environment variable
    3. Auto-detection
    """
    global _forced_profile
    if _forced_profile is not None:
        return _forced_profile

    env_name = os.environ.get("ZIPLINE_HARDWARE_PROFILE", "").strip()
    if env_name:
        profile = detect_profile()
        if env_name != profile.name:
            logger.info(
                "ZIPLINE_HARDWARE_PROFILE=%s requested but detected %s; "
                "using detected profile",
                env_name, profile.name,
            )
        return profile

    return detect_profile()


def set_profile(profile: HardwareProfile) -> None:
    """Force a specific hardware profile (useful for testing)."""
    global _forced_profile
    _forced_profile = profile


def reset_profile() -> None:
    """Clear any forced profile, reverting to auto-detection."""
    global _forced_profile
    _forced_profile = None
    detect_profile.cache_clear()
