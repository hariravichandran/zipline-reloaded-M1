"""Platform-specific runtime configuration for optimal performance.

This module applies hardware-aware settings at import time based on the
detected hardware profile.  It is imported early in zipline's startup
sequence (from ``zipline/__init__.py``) to ensure settings are active
before any computation begins.

On Apple Silicon (M-series) Macs, this:
- Configures blosc compression threads for bcolz I/O
- Sets optimal numexpr thread count to avoid contention with Accelerate
- Tunes numpy/scipy threading via environment variables

On Linux x86_64:
- Configures OpenBLAS/MKL thread counts
- Sets blosc threads based on physical core count
"""

import logging
import os

logger = logging.getLogger(__name__)

_configured = False


def configure_for_platform():
    """Apply platform-aware performance settings.

    Safe to call multiple times — only runs once.
    """
    global _configured
    if _configured:
        return
    _configured = True

    try:
        from zipline.utils.hardware_profile import get_profile
        profile = get_profile()
    except Exception:
        return

    _configure_blosc(profile)
    _configure_threading(profile)

    logger.debug(
        "Platform config applied: %s (threads=%d, blosc=%d)",
        profile.name,
        profile.optimal_threads,
        profile.blosc_nthreads,
    )


def _configure_blosc(profile):
    """Set blosc thread count for bcolz decompression."""
    try:
        import blosc
        blosc.set_nthreads(profile.blosc_nthreads)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("blosc thread config failed: %s", e)


def _configure_threading(profile):
    """Set thread counts for BLAS/LAPACK and numexpr.

    On M-series Macs, Apple Accelerate manages its own threading, but we
    limit numexpr to the performance core count to avoid contention.

    On Linux, we set OpenBLAS/MKL thread counts via environment variables
    (must be done before numpy is imported for full effect, but setting them
    here still affects some libraries).
    """
    threads = str(profile.optimal_threads)

    if profile.name == "apple_silicon":
        # Apple Accelerate automatically uses the optimal number of threads.
        # Limit numexpr to perf cores to avoid contention.
        # VECLIB_MAXIMUM_THREADS is respected by Accelerate.
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", threads)
        try:
            import numexpr
            numexpr.set_num_threads(profile.optimal_threads)
        except (ImportError, AttributeError):
            pass

    elif profile.name == "linux_x86":
        # These env vars control OpenBLAS and MKL threading.
        # Setting them here may not affect already-loaded libraries, but it
        # will affect any that are loaded later (e.g., scipy).
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
        try:
            import numexpr
            numexpr.set_num_threads(profile.optimal_threads)
        except (ImportError, AttributeError):
            pass

    elif profile.name == "linux_arm":
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
        try:
            import numexpr
            numexpr.set_num_threads(profile.optimal_threads)
        except (ImportError, AttributeError):
            pass
