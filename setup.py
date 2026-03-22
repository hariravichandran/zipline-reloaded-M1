#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup  # noqa: E402


# --- Architecture-aware compiler flags ---
def _get_extra_compile_args():
    """Return compiler flags optimised for the current platform.

    * Apple Silicon (arm64 / macOS): ``-mcpu=apple-m1`` lets clang emit
      code tuned for the Apple M-series performance cores while remaining
      compatible with M1 through M4.  ``-O3`` enables aggressive
      optimisation (loop vectorisation, auto-NEON, …).
    * x86_64 Linux / macOS: ``-O3 -march=native`` for best performance
      on the build host.
    * Fallback: ``-O3`` only.
    """
    machine = platform.machine().lower()
    system = platform.system().lower()

    if machine in ("arm64", "aarch64"):
        if system == "darwin":
            # Apple Silicon – apple-m1 target covers M1–M4
            return ["-O3", "-mcpu=apple-m1"]
        # Generic ARM64 (Linux aarch64, Graviton, etc.)
        return ["-O3", "-mcpu=native"]
    if machine in ("x86_64", "amd64"):
        return ["-O3", "-march=native"]
    return ["-O3"]


_EXTRA_COMPILE_ARGS = _get_extra_compile_args()

# Common macros for all extensions
_COMMON_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


def _make_extension(name, sources, depends=None):
    """Helper to create an Extension with shared compiler settings."""
    kw = dict(
        name=name,
        sources=sources,
        define_macros=list(_COMMON_MACROS),
        extra_compile_args=list(_EXTRA_COMPILE_ARGS),
    )
    if depends:
        kw["depends"] = depends
    return Extension(**kw)


def window_specialization(typename):
    """Make an extension for an AdjustedArrayWindow specialization."""
    return _make_extension(
        name=f"zipline.lib._{typename}window",
        sources=[f"src/zipline/lib/_{typename}window.pyx"],
        depends=["src/zipline/lib/_windowtemplate.pxi"],
    )


ext_options = dict(
    compiler_directives=dict(profile=True, language_level="3"),
    annotate=True,
)
ext_modules = [
    _make_extension("zipline.assets._assets",
                    ["src/zipline/assets/_assets.pyx"]),
    _make_extension("zipline.assets.continuous_futures",
                    ["src/zipline/assets/continuous_futures.pyx"]),
    _make_extension("zipline.lib.adjustment",
                    ["src/zipline/lib/adjustment.pyx"]),
    _make_extension("zipline.lib._factorize",
                    ["src/zipline/lib/_factorize.pyx"]),
    window_specialization("float64"),
    window_specialization("int64"),
    window_specialization("int64"),
    window_specialization("uint8"),
    window_specialization("label"),
    _make_extension("zipline.lib.rank",
                    ["src/zipline/lib/rank.pyx"]),
    _make_extension("zipline.data._equities",
                    ["src/zipline/data/_equities.pyx"]),
    _make_extension("zipline.data._adjustments",
                    ["src/zipline/data/_adjustments.pyx"]),
    _make_extension("zipline._protocol",
                    ["src/zipline/_protocol.pyx"]),
    _make_extension("zipline.finance._finance_ext",
                    ["src/zipline/finance/_finance_ext.pyx"]),
    _make_extension("zipline.gens.sim_engine",
                    ["src/zipline/gens/sim_engine.pyx"]),
    _make_extension("zipline.data._minute_bar_internal",
                    ["src/zipline/data/_minute_bar_internal.pyx"]),
    _make_extension("zipline.data._resample",
                    ["src/zipline/data/_resample.pyx"]),
]

setup(
    use_scm_version=True,
    ext_modules=cythonize(ext_modules, **ext_options),
    include_dirs=[numpy.get_include()],
)
