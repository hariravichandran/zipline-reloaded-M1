# Zipline Reloaded (ARM / Apple Silicon Fork)

## Overview
This is a fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) that runs **natively on Apple Silicon (M1–M4)** and ARM64 Linux — no Rosetta or x86 emulation required. Full x86_64 Linux/macOS/Windows compatibility is retained.

## Key Information
- **Python Version**: >= 3.10
- **Main Dependencies**: pandas >= 2.0, SQLAlchemy >= 2.0, numpy >= 2.0
- **Primary Target**: macOS arm64 (Apple Silicon M1–M4)
- **Also Supports**: x86_64 Linux, x86_64 macOS, aarch64 Linux, Windows x64
- **Upstream**: https://github.com/stefan-jansen/zipline-reloaded

## ARM-Specific Changes (vs upstream)
1. **`setup.py`**: Architecture-aware compiler flags (`-mcpu=apple-m1` on Apple Silicon, `-mcpu=native` on aarch64 Linux, `-march=native` on x86_64) for optimised Cython extension builds.
2. **`pyproject.toml`**: Updated cibuildwheel config to target arm64 first; `MACOSX_DEPLOYMENT_TARGET=11.0` (minimum for arm64); added `aarch64` to Linux build targets.
3. **CI workflows**: All macOS jobs use `macos-14` runners (Apple Silicon M1). Dedicated `ci_arm64.yml` workflow validates ARM builds.
4. **`tools/verify_arm_build.py`**: Script to verify a build is correctly running on ARM64 (checks .so architecture, Rosetta detection, all Cython extensions).

## Project Structure
- `src/zipline/`: Main source code
  - `algorithm.py`: Core algorithm execution
  - `api.py`: Public API functions
  - `data/`: Data ingestion and handling (bcolz daily/minute bars, HDF5)
  - `finance/`: Financial calculations and order execution
  - `pipeline/`: Factor-based screening system
  - `lib/`: Cython extensions (adjustment, rank, factorize, window specializations)
- `tests/`: Test suite
- `tools/`: Build utilities and verification scripts
- `docs/`: Documentation source

## Development Commands
```bash
# Install in development mode (ARM-native)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# Verify ARM build
python tools/verify_arm_build.py

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_algorithm.py

# Build documentation
cd docs && make html
```

## Build Notes
- The project uses Cython for 16 performance-critical C extensions
- All extensions build natively on ARM64 — no patches needed
- `bcolz-zipline` has ARM64 wheels on PyPI for Python 3.10–3.13
- `tables`, `h5py`, `scipy`, and all other C-extension deps have ARM64 wheels
- On macOS ARM, the build uses `-mcpu=apple-m1 -O3` for optimal codegen
- On x86_64, the build uses `-march=native -O3`

## Testing Approach
- Unit tests use pytest with pytest-xdist for parallel execution
- Test data is stored in `tests/resources/`
- CI runs on `macos-14` (Apple Silicon), `ubuntu-latest`, and `windows-latest`

## Important Notes
- Be careful with numpy/pandas API changes due to major version updates
- Trading calendars are handled by the external `exchange_calendars` package
- When syncing with upstream, watch for changes to `setup.py` compiler flags
