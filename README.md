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

This is a fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) with native Apple Silicon support, performance optimizations, and bug fixes. It runs on **all platforms** the original supports, plus ARM64 natively.

| | |
| --- | --- |
| **PyPI** | `pip install zipline-reloaded-m1` |
| **Python** | >= 3.10 |
| **Platforms** | macOS arm64 (M1-M4), macOS x86_64, Linux x86_64, Linux aarch64, Windows x64 |
| **Upstream** | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |

---

## Improvements Over Upstream

### Native ARM64 / Apple Silicon Support
- **Zero Rosetta dependency**: All 16 Cython extensions compile and run natively on arm64. No x86 emulation layer needed.
- **Architecture-aware compiler flags**: `setup.py` auto-detects the platform and applies optimal flags:
  - Apple Silicon (M1-M4): `-mcpu=apple-m1 -O3` — tuned for Apple's performance cores
  - Linux aarch64 (Graviton, Ampere): `-mcpu=native -O3`
  - x86_64 (Intel/AMD): `-march=native -O3`
- **ARM64 CI validation**: Dedicated `ci_arm64.yml` workflow runs on GitHub's `macos-14` Apple Silicon runners for Python 3.10–3.13.
- **ARM build verification tool**: `python tools/verify_arm_build.py` checks native execution (no Rosetta), .so architecture, all Cython extensions, and key dependencies.

### Performance Optimizations
- **Cython compiler directives (global)**: All extensions now build with `boundscheck=False`, `wraparound=False`, `cdivision=True`, and `initializedcheck=False` by default. This eliminates per-access safety checks in the inner loops of backtesting.
- **Profiling disabled in production**: Upstream builds with `profile=True` which adds overhead to every Cython function call. We default to `profile=False` (set `ZIPLINE_DEBUG=1` to re-enable for development).
- **Parallel Cython compilation**: Extensions compile using all available CPU cores (`nthreads=cpu_count()`).
- **Hot-path decorators**: Added `@boundscheck(False)`, `@wraparound(False)`, and `@cdivision(True)` to performance-critical functions in `_minute_bar_internal.pyx` and `_equities.pyx` that run on every bar during backtests.

### Bug Fixes
- **pandas 2.3 compatibility**: Fixed `TypeError: only 0-dimensional arrays can be converted to Python scalars` in EWM-based pipeline factors (EWMA, EWMSTD, MACD). The upstream tests fail on pandas >= 2.3; this fork works correctly.
- **Full test suite passing**: 3,159 tests pass on ARM64 (0 failures, 17 skipped).

### CI/CD Improvements
- **All macOS CI jobs use Apple Silicon** (`macos-14`) instead of x86 runners.
- **Linux aarch64 build verification** via QEMU emulation in CI.
- **Wheel builds**: `arm64` is the primary macOS target, with `MACOSX_DEPLOYMENT_TARGET=11.0` (minimum for arm64 support).

---

## Features

- **Ease of Use:** Zipline tries to get out of your way so that you can focus on algorithm development. See below for a code example.
- **Batteries Included:** many common statistics like moving average and linear regression can be readily accessed from within a user-written algorithm.
- **PyData Integration:** Input of historical data and output of performance statistics are based on Pandas DataFrames to integrate nicely into the existing PyData ecosystem.
- **Statistics and Machine Learning Libraries:** You can use libraries like matplotlib, scipy, statsmodels, and scikit-learn to support development, analysis, and visualization of state-of-the-art trading systems.

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

### Linux / Windows

```bash
pip install zipline-reloaded-m1
```

All dependencies (bcolz-zipline, tables, h5py, scipy, numpy, pandas) have pre-built wheels for all supported platforms.

### Using `conda`

If you are using [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html), the upstream `zipline-reloaded` is available on `conda-forge`:

```bash
conda install -c conda-forge zipline-reloaded
```

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
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

    # Trading logic
    if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)
```

You can then run this algorithm using the Zipline CLI. But first, you need to download some market data with historical prices and trading volumes.

This will download asset pricing data from [NASDAQ](https://data.nasdaq.com/databases/WIKIP) (formerly [Quandl](https://www.nasdaq.com/about/press-center/nasdaq-acquires-quandl-advance-use-alternative-data)).

> This requires an API key, which you can get for free by signing up at [NASDAQ Data Link](https://data.nasdaq.com).

```bash
$ export QUANDL_API_KEY="your_key_here"
$ zipline ingest -b quandl
````

The following will
- stream the through the algorithm over the specified time range.
- save the resulting performance DataFrame as `dma.pickle`, which you can load and analyze from Python using, e.g., [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded).

```bash
$ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark
```

You can find other examples in the [zipline/examples](https://github.com/hariravichandran/zipline-reloaded-M1/tree/main/src/zipline/examples) directory.

## Questions, suggestions, bugs?

If you find a bug or have other questions about the library, feel free to [open an issue](https://github.com/hariravichandran/zipline-reloaded-M1/issues/new).
