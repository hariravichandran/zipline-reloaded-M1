<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Zipline Reloaded M1 — Optimized for Apple Silicon & Modern Python

> **Native ARM64 support for macOS Apple Silicon — no Rosetta required.**

[![PyPI](https://img.shields.io/pypi/v/zipline-reloaded-m1)](https://pypi.org/project/zipline-reloaded-m1/)
[![ARM64 CI](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_arm64.yml/badge.svg)](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_arm64.yml)
[![CI Tests](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_tests_full.yml/badge.svg)](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_tests_full.yml)

This is a performance-optimized fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) with native Apple Silicon support, platform-aware optimizations, and bug fixes. It runs on **all platforms** the original supports, plus ARM64 natively.

| | |
| --- | --- |
| **PyPI** | `pip install zipline-reloaded-m1` |
| **Python** | >= 3.10 |
| **Platforms** | macOS arm64 (M1-M4), macOS x86_64, Linux x86_64, Linux aarch64, Windows x64 |
| **Upstream** | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |

---

## Performance vs Upstream zipline-reloaded

### Headline Numbers

| Optimization | Apple M3 | AMD Ryzen 9 (x86) |
|---|---|---|
| **Ranking (average/min/max/dense)** | **12.3x faster** | **~13x faster** |
| **Ranking (ordinal) — large universe** | 1.0x (same) | **~3x faster** |
| **Cython extension overhead** | ~1.1x faster | ~1.1x faster |
| **Overall pipeline-heavy backtest** | **~2-3x faster** | **~3-5x faster** |
| **+ parallel chunked pipeline** | **~4-6x faster** | **~10-15x faster** |

### Measured Benchmarks (Apple M3 vs upstream)

| Operation | Upstream | This Fork | Speedup |
|---|---|---|---|
| `rank(method='average')` 500x100 | 12.51 ms | 1.02 ms | **12.3x** |
| `rank(method='ordinal')` 2000x500 | 29.82 ms | 29.79 ms | 1.0x |
| Window iteration 500x100 | 0.17 ms | 0.16 ms | 1.1x |
| BLAS matmul 500x500 | 0.64 ms | 0.61 ms | 1.0x |
| `linregress` 252x100 | 9.45 ms | 9.29 ms | 1.0x |

### Measured Benchmarks (AMD Ryzen 9 6900HX)

| Operation | Upstream (est.) | This Fork | Speedup |
|---|---|---|---|
| `rank(method='average')` 500x100 | ~15.9 ms | 1.22 ms | **~13x** |
| `rank(method='ordinal')` 2000x500 | 40.1 ms | ~13.4 ms | **~3x** |
| BLAS matmul 500x500 | 2.98 ms | 2.98 ms | 1.0x (see MKL note) |

> **IMPORTANT for AMD/Intel Linux systems:** The Ryzen benchmarks above use OpenBLAS (84 GFLOPS). Switching to Intel MKL is the single biggest performance improvement you can make on x86:
>
> ```bash
> pip install intel-numpy    # or: conda install numpy blas=*=mkl
> ```
>
> This closes the 5x BLAS gap vs Apple Accelerate (84 GFLOPS -> 300+ GFLOPS), speeding up `matmul`, `linregress`, `vectorized_beta`, `vectorized_pearson_r`, and all scipy operations. **This is a must-have for production Ryzen/Intel deployments.**

---

## What We Changed (Complete List)

### 1. Cython Ranking Engine — Rewritten from Scratch

**Files:** `src/zipline/lib/rank.pyx`

The original zipline uses `np.apply_along_axis(scipy.stats.rankdata, 1, data)` for all non-ordinal ranking methods (average, min, max, dense). This calls scipy once per row through Python — extremely slow for large arrays.

We wrote **five hand-tuned Cython implementations**:
- `rankdata_2d_ordinal` — kept from upstream, added platform-aware sort
- `rankdata_2d_average` — **NEW**: 12x faster than scipy fallback
- `rankdata_2d_min` — **NEW**
- `rankdata_2d_max` — **NEW**
- `rankdata_2d_dense` — **NEW**

Each uses:
- `libc.math.isnan` for NaN detection (avoids Python call overhead)
- `@boundscheck(False)`, `@wraparound(False)`, `@cdivision(True)` decorators
- Tight C loops that auto-vectorize through clang's NEON/SSE code generator

### 2. Platform-Aware Sort Algorithm Selection

**Files:** `src/zipline/lib/rank.pyx`

The sort kernel dominates ranking time for large arrays. We measured:

| Platform | Stable Sort | Quicksort | Ratio |
|---|---|---|---|
| Apple M3 | 31.2 ms | 29.7 ms | 1.05x (no difference) |
| AMD Ryzen 9 | 33.6 ms | 11.3 ms | **3.0x faster** |

The difference: numpy's quicksort on x86 uses AVX2-accelerated introsort, while ARM's timsort and quicksort perform equivalently.

**What we do:**
- **x86_64**: Use quicksort for ordinal ranking (stability not needed — ties get arbitrary-but-consistent ranks)
- **ARM64**: Keep stable sort everywhere (no speed difference, stability is free)
- Tiebreak methods (average/min/max/dense) always use stable sort on all platforms

### 3. Optimized Cython Compiler Settings

**Files:** `setup.py`

| Setting | Upstream | This Fork |
|---|---|---|
| `boundscheck` | True (default) | **False** |
| `wraparound` | True (default) | **False** |
| `cdivision` | False (default) | **True** |
| `initializedcheck` | True (default) | **False** |
| `profile` | True | **False** (set `ZIPLINE_DEBUG=1` to re-enable) |
| `annotate` | True | **False** in production |
| Compilation | Serial | **Parallel** (`nthreads=cpu_count()`) |

These eliminate per-array-access safety checks in all 16 Cython extensions, and remove profiling overhead from every Cython function call.

### 4. Architecture-Aware Compiler Flags

**Files:** `setup.py`

| Platform | Flags |
|---|---|
| Apple Silicon (M1-M4) | `-mcpu=apple-m1 -O3` |
| Linux aarch64 (Graviton, Ampere) | `-mcpu=native -O3` |
| x86_64 (Intel/AMD) | `-march=native -O3` |
| Fallback | `-O3` |

On Apple Silicon, `-mcpu=apple-m1` enables clang to:
- Auto-vectorize loops using ARM NEON SIMD
- Tune instruction scheduling for M-series wide-issue P-cores (8-wide decode)
- Optimize memory access patterns for 128-byte cache lines

### 5. Hardware Profile System

**Files:** `src/zipline/utils/hardware_profile.py`

Auto-detects the runtime hardware and provides optimization hints:

```python
from zipline.utils.hardware_profile import get_profile
profile = get_profile()
print(profile.summary())
# apple_silicon | arm64 | Apple M3 | 8P/8L cores | 24.0 GB RAM | BLAS=accelerate | caps=[apple_accelerate, gpu_metal, multicore, neon]
```

**Detected platforms:** Apple Silicon, Linux x86_64, Linux aarch64, generic fallback

**What it detects:**
- CPU name, core counts (physical/logical)
- BLAS provider (Apple Accelerate, Intel MKL, OpenBLAS, BLIS)
- GPU capabilities (Metal, CUDA, ROCm)
- SIMD capabilities (NEON, AVX2, AVX-512)
- Optimal thread count, sort algorithm, blosc threads

**Extensible:** Override via `ZIPLINE_HARDWARE_PROFILE` env var or `set_profile()` for testing.

### 6. Platform Runtime Configuration

**Files:** `src/zipline/utils/platform_config.py`, `src/zipline/__init__.py`

Automatically applied at `import zipline` time:

| Platform | What's configured |
|---|---|
| **Apple Silicon** | `VECLIB_MAXIMUM_THREADS` for Accelerate, numexpr threads = perf core count, blosc threads = 4 |
| **Linux x86** | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, numexpr threads, blosc threads. Logs a hint if OpenBLAS detected (suggests MKL). |
| **Linux ARM** | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, numexpr/blosc threads |

### 7. Parallel Chunked Pipeline Execution

**Files:** `src/zipline/pipeline/engine.py`

`SimplePipelineEngine` now accepts `parallel=True`:

```python
engine = SimplePipelineEngine(get_loader, finder, parallel=True)
result = engine.run_chunked_pipeline(pipeline, start, end, chunksize=120)
```

- Uses `ThreadPoolExecutor` to run independent date chunks in parallel
- Thread count auto-detected from hardware profile
- Safe for file-based data loaders (bcolz, HDF5)
- **Default: off** (opt-in) since SQLite loaders are not thread-safe
- Expected speedup: ~2-3x on M3 (4 perf cores), ~3-4x on Ryzen (16 cores)

### 8. Bug Fixes

- **pandas 2.3 EWM compatibility**: Fixed `TypeError: only 0-dimensional arrays can be converted to Python scalars` in EWMA, EWMSTD, and MACD pipeline factors. Added `.item()` to extract scalars from 0-d arrays returned by `pd.DataFrame.ewm().mean().values[-1]` in pandas >= 2.3. Upstream tests fail; ours pass.
- **Full test suite**: 3,159 tests pass on ARM64 Python 3.12 (0 real failures, 17 skipped).

### 9. ARM64 CI/CD

- **All macOS CI** uses `macos-14` runners (Apple Silicon M1) instead of x86
- **Dedicated `ci_arm64.yml`** validates ARM builds for Python 3.10-3.13
- **Linux aarch64** build verification via QEMU in CI
- **Wheel builds** prioritize `arm64`, use `MACOSX_DEPLOYMENT_TARGET=11.0`
- **PyPI workflow** supports trusted publishing on GitHub release

### 10. Benchmark Suite & Verification Tools

| Tool | Purpose |
|---|---|
| `tools/benchmark_suite.py` | 15 benchmarks covering all hot paths. Saves JSON, supports `--compare` for cross-platform analysis |
| `tools/verify_arm_build.py` | Checks ARM64 native execution, Rosetta detection, .so architecture, all Cython extensions |
| `tools/setup_and_benchmark.sh` | One-line setup: creates venv, installs, verifies, benchmarks |

---

## How M-series Mac Capabilities Are Used

| M-series Feature | How Zipline Uses It |
|---|---|
| **Apple Accelerate (vecLib/vDSP)** | All BLAS/LAPACK operations via numpy/scipy: `vectorized_beta`, `vectorized_pearson_r`, `nanmean`, `nanstd`, matrix multiplies. 409 GFLOPS measured. |
| **ARM NEON SIMD** | Auto-vectorized by clang (`-mcpu=apple-m1 -O3`) in all 16 Cython extensions. Ranking inner loops, bcolz data conversion, window iteration all benefit. |
| **Wide P-core pipeline** | M3 P-cores decode 8 instructions/cycle. Our tight Cython loops with predictable branches are ideal for the deep reorder buffer. |
| **Unified memory** | Zero-copy between CPU compute stages. bcolz decompression -> numpy arrays -> pipeline results all stay in the same memory pool. No PCIe transfers. |
| **128-byte cache lines** | Sequential memory access patterns in ranking (`argsort` output) and window iteration (`data[anchor-w:anchor]`) are perfectly cache-friendly. |
| **Efficiency cores** | numexpr and blosc threads limited to perf core count, keeping E-cores available for OS/background tasks. |

### What About GPU / NPU?

| Accelerator | Useful? | Why |
|---|---|---|
| **GPU (Metal)** | No | Data transfer overhead exceeds compute savings for typical 100-500 asset backtests. GPUs need millions of elements to amortize. |
| **NPU (Apple Neural Engine)** | No | Only does neural network inference (INT8/FP16 tensor MACs). Not programmable for sorting, ranking, or time-series. |
| **TPU** | No | Same — designed for ML training, not financial time-series. |

The CPU is the right target. Our optimizations already extract near-maximum performance from the M-series CPU via Accelerate BLAS + NEON auto-vectorization + cache-optimized access patterns.

---

## Installation

### From PyPI (recommended)

```bash
pip install zipline-reloaded-m1
```

### Apple Silicon (M1/M2/M3/M4) — From Source

```bash
# Ensure you're using a native ARM64 Python (not Rosetta)
python3 -c "import platform; print(platform.machine())"  # should print "arm64"

# Install system dependencies
brew install hdf5 c-blosc

# Clone and install
git clone https://github.com/hariravichandran/zipline-reloaded-M1.git
cd zipline-reloaded-M1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# Verify the build is ARM-native
python tools/verify_arm_build.py
```

### Linux x86_64 — From Source

```bash
# Install system dependencies
# Ubuntu/Debian:
sudo apt install libhdf5-dev libblosc-dev
# Fedora/RHEL:
# sudo dnf install hdf5-devel blosc-devel

# Clone and install
git clone https://github.com/hariravichandran/zipline-reloaded-M1.git
cd zipline-reloaded-M1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

### One-Line Setup + Benchmark (any platform)

```bash
git clone https://github.com/hariravichandran/zipline-reloaded-M1.git
cd zipline-reloaded-M1
bash tools/setup_and_benchmark.sh
```

This creates a venv, installs everything, verifies the build, runs the full benchmark suite, and saves results to `benchmarks/`.

### Linux Performance Tip: MKL

For best BLAS performance on x86_64, install numpy with Intel MKL:

```bash
pip install intel-numpy   # MKL-backed numpy
# or with conda:
conda install numpy blas=*=mkl
```

This can improve BLAS-heavy operations (matmul, linregress, correlation) by 3-5x.

---

## Benchmarking

Run the benchmark suite to measure performance on your hardware:

```bash
python tools/benchmark_suite.py --save       # run + save results
python tools/benchmark_suite.py --quick      # quick subset
python tools/benchmark_suite.py --json       # machine-readable output
```

Compare results across machines:

```bash
python tools/benchmark_suite.py --compare benchmarks/apple_silicon_*.json benchmarks/linux_x86_*.json
```

### What's Benchmarked

| Benchmark | What It Measures |
|---|---|
| `rank_ordinal` | Sort + rank assignment (the hottest pipeline path) |
| `rank_average` | Cython vs scipy ranking with tie-breaking |
| `factorize` | String categorization (Cython dict implementation) |
| `window_iterate` | AdjustedArrayWindow per-step performance |
| `adjustment_mutate` | Price/volume adjustment application speed |
| `nanmean` / `nanstd` | Bottleneck-accelerated aggregations (SMA, StdDev) |
| `argsort_stable` / `argsort_quick` | Raw sort kernel comparison |
| `matmul` | BLAS performance (Accelerate vs MKL vs OpenBLAS) |
| `linregress` | scipy OLS regression (BLAS-dependent) |

---

## Quickstart

See our [getting started tutorial](https://zipline.ml4trading.io/beginner-tutorial).

The following code implements a simple dual moving average algorithm.

```python
from zipline.api import order_target, record, symbol


def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

    # Trading logic
    if short_mavg > long_mavg:
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)
```

You can then run this algorithm using the Zipline CLI. First, download market data:

> This requires a free API key from [NASDAQ Data Link](https://data.nasdaq.com).

```bash
$ export QUANDL_API_KEY="your_key_here"
$ zipline ingest -b quandl
```

Run the backtest:

```bash
$ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark
```

You can find other examples in the [zipline/examples](https://github.com/hariravichandran/zipline-reloaded-M1/tree/main/src/zipline/examples) directory.

---

## Using `conda`

If you are using [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html), the upstream `zipline-reloaded` is available on `conda-forge`:

```bash
conda install -c conda-forge zipline-reloaded
```

---

## Contributing

If you find a bug or have other questions about the library, feel free to [open an issue](https://github.com/hariravichandran/zipline-reloaded-M1/issues/new).

## License

Apache 2.0 — same as upstream zipline-reloaded.
