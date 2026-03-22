#!/usr/bin/env bash
# =============================================================================
# Quick setup + benchmark for a fresh machine (Linux or macOS)
#
# Usage:
#   git clone https://github.com/hariravichandran/zipline-reloaded-M1.git
#   cd zipline-reloaded-M1
#   bash tools/setup_and_benchmark.sh
#
# What this does:
#   1. Creates a Python venv
#   2. Installs zipline-reloaded-m1 with test dependencies
#   3. Runs the ARM build verification (macOS only)
#   4. Runs the full benchmark suite and saves results
#   5. Prints comparison instructions
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================="
echo "  Zipline Reloaded M1 — Setup & Benchmark"
echo "============================================="
echo ""

# --- Detect Python ---
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python >= 3.10 not found."
    echo "Install Python 3.10+ and try again."
    exit 1
fi

echo "Using Python: $($PYTHON --version) at $(which $PYTHON)"
echo "Architecture: $($PYTHON -c 'import platform; print(platform.machine())')"
echo ""

# --- Install system dependencies (if needed) ---
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    echo "macOS detected — checking for hdf5 and c-blosc..."
    if ! brew list hdf5 &>/dev/null; then
        echo "Installing hdf5 via Homebrew..."
        brew install hdf5
    fi
    if ! brew list c-blosc &>/dev/null; then
        echo "Installing c-blosc via Homebrew..."
        brew install c-blosc
    fi
elif [ "$OS" = "Linux" ]; then
    echo "Linux detected — checking for HDF5 dev headers..."
    if ! pkg-config --exists hdf5 2>/dev/null; then
        echo ""
        echo "NOTE: You may need HDF5 and blosc dev libraries."
        echo "  Ubuntu/Debian: sudo apt install libhdf5-dev libblosc-dev"
        echo "  Fedora/RHEL:   sudo dnf install hdf5-devel blosc-devel"
        echo ""
    fi
fi

# --- Create venv ---
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

echo "Activating venv..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- Install zipline ---
echo ""
echo "Installing zipline-reloaded-m1 (this may take a few minutes)..."
pip install -e ".[test]" -q

# --- Verify build ---
if [ "$OS" = "Darwin" ]; then
    echo ""
    echo "Running ARM build verification..."
    python tools/verify_arm_build.py
fi

# --- Run benchmarks ---
echo ""
echo "============================================="
echo "  Running Benchmark Suite"
echo "============================================="
echo ""
python tools/benchmark_suite.py --save

echo ""
echo "============================================="
echo "  Done!"
echo "============================================="
echo ""
echo "Benchmark results saved to benchmarks/ directory."
echo ""
echo "To compare with other machines:"
echo "  1. Commit and push: git add benchmarks/ && git commit -m 'bench: add <machine> results' && git push"
echo "  2. Or copy the JSON file and compare manually:"
echo "     python tools/benchmark_suite.py --compare benchmarks/apple_silicon_*.json benchmarks/linux_x86_*.json"
echo ""
echo "To run tests:  python -m pytest tests/ --timeout=300 -q"
echo "To re-run benchmarks: python tools/benchmark_suite.py --save"
