#!/usr/bin/env python3
"""Verify that zipline-reloaded is correctly built for ARM64 (Apple Silicon).

Run after installing:
    python tools/verify_arm_build.py

Exits 0 if everything is OK, 1 if any check fails.
"""

import platform
import struct
import subprocess
import sys


def _header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def check_architecture():
    """Verify we're running on ARM64 natively (no Rosetta)."""
    _header("Architecture check")
    machine = platform.machine()
    print(f"  platform.machine() = {machine}")
    print(f"  pointer size       = {struct.calcsize('P') * 8}-bit")
    print(f"  sys.platform       = {sys.platform}")

    if machine not in ("arm64", "aarch64"):
        print(f"  WARNING: Not running on ARM64 (got {machine})")
        print("  This script is designed for Apple Silicon / aarch64")
        return False

    # On macOS, check we're not under Rosetta
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "sysctl.proc_translated"],
                capture_output=True, text=True
            )
            translated = result.stdout.strip()
            if translated == "1":
                print("  ERROR: Running under Rosetta 2 (x86_64 emulation)!")
                print("  Ensure you're using a native ARM64 Python.")
                return False
            print("  Native ARM64 execution confirmed (no Rosetta)")
        except FileNotFoundError:
            pass

    return True


def check_python():
    """Verify Python itself is ARM64."""
    _header("Python binary check")
    executable = sys.executable
    print(f"  Python: {sys.version}")
    print(f"  Executable: {executable}")

    if sys.platform == "darwin":
        result = subprocess.run(
            ["file", executable], capture_output=True, text=True
        )
        output = result.stdout
        print(f"  file output: {output.strip()}")
        if "arm64" in output:
            print("  Python binary is ARM64")
            return True
        elif "x86_64" in output and "arm64" not in output:
            print("  ERROR: Python binary is x86_64 only!")
            return False
    return True


def check_cython_extensions():
    """Verify all Cython extensions import correctly."""
    _header("Cython extension import check")

    extensions = [
        ("zipline.lib.adjustment", "Float64Multiply"),
        ("zipline.lib.rank", "rankdata_2d_ordinal"),
        ("zipline.lib._factorize", None),
        ("zipline.lib._float64window", None),
        ("zipline.lib._int64window", None),
        ("zipline.lib._uint8window", None),
        ("zipline.lib._labelwindow", None),
        ("zipline.assets._assets", "Asset"),
        ("zipline.assets.continuous_futures", None),
        ("zipline._protocol", "BarData"),
        ("zipline.finance._finance_ext", None),
        ("zipline.gens.sim_engine", "MinuteSimulationClock"),
        ("zipline.data._equities", None),
        ("zipline.data._adjustments", None),
        ("zipline.data._minute_bar_internal", None),
        ("zipline.data._resample", None),
    ]

    all_ok = True
    for module_name, attr_name in extensions:
        try:
            mod = __import__(module_name, fromlist=[attr_name or ""])
            if attr_name:
                getattr(mod, attr_name)
            print(f"  OK  {module_name}" + (f".{attr_name}" if attr_name else ""))
        except Exception as e:
            print(f"  FAIL  {module_name}: {e}")
            all_ok = False

    return all_ok


def check_so_architecture():
    """Verify .so files are compiled for ARM64."""
    _header("Shared object architecture check")

    if sys.platform != "darwin":
        print("  Skipping (macOS-specific check)")
        return True

    import zipline.lib.rank as rank_mod
    so_path = rank_mod.__file__
    print(f"  Sample .so: {so_path}")

    result = subprocess.run(
        ["file", so_path], capture_output=True, text=True
    )
    output = result.stdout
    print(f"  file output: {output.strip()}")

    if "arm64" in output:
        print("  .so is compiled for ARM64")
        return True
    else:
        print("  WARNING: .so may not be ARM64 native")
        return False


def check_key_dependencies():
    """Verify key dependencies that historically had ARM issues."""
    _header("Key dependency check")

    deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("bcolz", "bcolz"),
        ("tables", "tables"),
        ("h5py", "h5py"),
        ("lru", "lru-dict"),
        ("bottleneck", "bottleneck"),
        ("numexpr", "numexpr"),
    ]

    all_ok = True
    for import_name, display_name in deps:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {display_name} {version}")
        except ImportError as e:
            print(f"  FAIL  {display_name}: {e}")
            all_ok = False

    return all_ok


def check_run_algorithm():
    """Verify the high-level API is importable."""
    _header("High-level API check")
    try:
        from zipline import run_algorithm
        from zipline.api import order, symbol, record
        print("  OK  run_algorithm, order, symbol, record all importable")
        return True
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


def main():
    print("Zipline ARM64 Build Verification")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")

    checks = [
        ("Architecture", check_architecture),
        ("Python binary", check_python),
        ("Cython extensions", check_cython_extensions),
        (".so architecture", check_so_architecture),
        ("Key dependencies", check_key_dependencies),
        ("High-level API", check_run_algorithm),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR during {name}: {e}")
            results[name] = False

    _header("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll checks passed! Zipline is correctly built for ARM64.")
    else:
        print("\nSome checks failed. See details above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
