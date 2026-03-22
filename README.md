<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Zipline Reloaded — Apple Silicon (M1/M2/M3/M4) Fork

> **Native ARM64 support for macOS Apple Silicon — no Rosetta required.**

This is a fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) that builds and runs **natively on Apple Silicon Macs** (M1, M2, M3, M4) and ARM64 Linux (aarch64). Full x86_64 compatibility is retained.

| | |
| --- | --- |
| **Upstream** | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |
| **Python** | >= 3.10 |
| **Platforms** | macOS arm64, macOS x86_64, Linux x86_64, Linux aarch64, Windows x64 |
| **CI** | [![ARM64 CI](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_arm64.yml/badge.svg)](https://github.com/hariravichandran/zipline-reloaded-M1/actions/workflows/ci_arm64.yml) |

## What's different from upstream?

- **ARM-optimised Cython builds**: `setup.py` detects the platform and uses `-mcpu=apple-m1 -O3` on Apple Silicon, `-mcpu=native -O3` on aarch64 Linux, or `-march=native -O3` on x86_64.
- **CI on Apple Silicon runners**: GitHub Actions `macos-14` (M1) runners for all macOS test jobs.
- **Wheel builds**: `cibuildwheel` configured with `arm64` as primary macOS target and `MACOSX_DEPLOYMENT_TARGET=11.0`.
- **Verification script**: `python tools/verify_arm_build.py` confirms native ARM execution, Rosetta detection, and .so architecture.
- **Linux aarch64**: Added to cibuildwheel and CI matrix for ARM server/Graviton support.

Zipline is a Pythonic event-driven system for backtesting, developed and used as the backtesting and live-trading engine by [crowd-sourced investment fund Quantopian](https://www.bizjournals.com/boston/news/2020/11/10/quantopian-shuts-down-cofounders-head-elsewhere.html). Since it closed late 2020, the domain that had hosted these docs expired. The library is used extensively in the book [Machine Learning for Algorithmic Trading](https://ml4trading.io)
by [Stefan Jansen](https://www.linkedin.com/in/applied-ai/) who is trying to keep the library up to date and available to his readers and the wider Python algotrading community.
- [Join our Community!](https://exchange.ml4trading.io)
- [Documentation](https://zipline.ml4trading.io)

## Features

- **Ease of Use:** Zipline tries to get out of your way so that you can focus on algorithm development. See below for a code example.
- **Batteries Included:** many common statistics like moving average and linear regression can be readily accessed from within a user-written algorithm.
- **PyData Integration:** Input of historical data and output of performance statistics are based on Pandas DataFrames to integrate nicely into the existing PyData ecosystem.
- **Statistics and Machine Learning Libraries:** You can use libraries like matplotlib, scipy, statsmodels, and scikit-klearn to support development, analysis, and visualization of state-of-the-art trading systems.

> **Note:** Release 3.05 makes Zipline compatible with Numpy 2.0, which requires Pandas 2.2.2 or higher. If you are using an older version of Pandas, you will need to upgrade it. Other packages may also still take more time to catch up with the latest Numpy release.

> **Note:** Release 3.0 updates Zipline to use [pandas](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v2.0.0.html) >= 2.0 and [SQLAlchemy](https://docs.sqlalchemy.org/en/20/) > 2.0. These are major version updates that may break existing code; please review the linked docs.

> **Note:** Release 2.4 updates Zipline to use [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars) >= 4.2. This is a major version update and may break existing code (which we have tried to avoid but cannot guarantee). Please review the changes [here](https://github.com/gerrymanoim/exchange_calendars/issues/61).

## Installation

Zipline supports Python >= 3.10 and is compatible with current versions of the relevant [NumFOCUS](https://numfocus.org/sponsored-projects?_sft_project_category=python-interface) libraries, including [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

### Apple Silicon (M1/M2/M3/M4) — Quick Start

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

### Using `pip` (from source)

```bash
pip install git+https://github.com/hariravichandran/zipline-reloaded-M1.git
```

### Using `conda`

If you are using the [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) distributions, you can install the upstream `zipline-reloaded` from the channel `conda-forge`:

```bash
conda install -c conda-forge zipline-reloaded
```

See the [installation](https://zipline.ml4trading.io/install.html) section of the docs for more detailed instructions.

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

You can find other examples in the [zipline/examples](https://github.com/stefan-jansen/zipline-reloaded/tree/main/src/zipline/examples) directory.

## Questions, suggestions, bugs?

If you find a bug or have other questions about the library, feel free to [open an issue](https://github.com/hariravichandran/zipline-reloaded-M1/issues/new).
