#!/usr/bin/env python3
"""Benchmark suite for zipline-reloaded performance analysis.

Measures the hot paths identified via profiling and reports results
alongside the detected hardware profile. Run on different machines to
compare performance across platforms.

Usage::

    python tools/benchmark_suite.py              # full suite
    python tools/benchmark_suite.py --quick      # quick smoke test
    python tools/benchmark_suite.py --json       # machine-readable output
    python tools/benchmark_suite.py --save       # save results to benchmarks/

Results are printed as a table and optionally saved to
``benchmarks/<profile_name>_<timestamp>.json`` for cross-platform comparison.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _time_fn(fn, n_runs=5, warmup=1):
    """Time a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
        "runs": n_runs,
    }


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_rank_ordinal_small():
    """rankdata_2d_ordinal on 500x100 array (typical daily pipeline)."""
    from zipline.lib.rank import rankdata_2d_ordinal
    data = np.random.randn(500, 100)
    return _time_fn(lambda: rankdata_2d_ordinal(data))


def bench_rank_ordinal_large():
    """rankdata_2d_ordinal on 2000x500 array (large universe)."""
    from zipline.lib.rank import rankdata_2d_ordinal
    data = np.random.randn(2000, 500)
    return _time_fn(lambda: rankdata_2d_ordinal(data), n_runs=3)


def bench_rank_average():
    """masked_rankdata_2d with method='average' (scipy fallback path)."""
    from zipline.lib.rank import masked_rankdata_2d
    data = np.random.randn(500, 100)
    mask = np.ones_like(data, dtype=bool)
    return _time_fn(
        lambda: masked_rankdata_2d(data, mask, np.nan, "average", True)
    )


def bench_factorize_strings():
    """factorize_strings on 10,000 random strings."""
    from zipline.lib._factorize import factorize_strings
    values = np.array(
        [f"SID_{i % 500}" for i in range(10000)], dtype=object
    )
    return _time_fn(lambda: factorize_strings(values, "", 1))


def bench_factorize_strings_large():
    """factorize_strings on 100,000 random strings."""
    from zipline.lib._factorize import factorize_strings
    values = np.array(
        [f"SID_{i % 2000}" for i in range(100000)], dtype=object
    )
    return _time_fn(lambda: factorize_strings(values, "", 1), n_runs=3)


def bench_window_iterate():
    """Float64 window iteration: 500 steps over 100-asset array, window=20."""
    from zipline.lib._float64window import AdjustedArrayWindow
    data = np.random.randn(520, 100).astype(np.float64)
    window = AdjustedArrayWindow(data, {}, {}, 0, 20, 0, None)
    def iterate():
        w = AdjustedArrayWindow(data, {}, {}, 0, 20, 0, None)
        for _ in w:
            pass
    return _time_fn(iterate)


def bench_window_iterate_large():
    """Float64 window iteration: 2000 steps over 500-asset array, window=20."""
    from zipline.lib._float64window import AdjustedArrayWindow
    data = np.random.randn(2020, 500).astype(np.float64)
    def iterate():
        w = AdjustedArrayWindow(data, {}, {}, 0, 20, 0, None)
        for _ in w:
            pass
    return _time_fn(iterate, n_runs=3)


def bench_adjustment_mutate():
    """Float64Multiply.mutate on 100x50 array (adjustment application)."""
    from zipline.lib.adjustment import Float64Multiply
    data = np.random.randn(100, 50).astype(np.float64)
    adj = Float64Multiply(0, 50, 0, 25, 1.05)
    def mutate():
        d = data.copy()
        adj.mutate(d)
    return _time_fn(mutate)


def bench_nanmean():
    """bottleneck nanmean on 252x500 array (SMA computation)."""
    try:
        from bottleneck import nanmean
    except ImportError:
        from numpy import nanmean
    data = np.random.randn(252, 500)
    data[data < -1.5] = np.nan  # ~7% NaN
    return _time_fn(lambda: nanmean(data, axis=0))


def bench_nanstd():
    """bottleneck nanstd on 252x500 array (StdDev computation)."""
    try:
        from bottleneck import nanstd
    except ImportError:
        from numpy import nanstd
    data = np.random.randn(252, 500)
    data[data < -1.5] = np.nan
    return _time_fn(lambda: nanstd(data, axis=0))


def bench_argsort_stable():
    """np.argsort (stable) on 2000x500 — sort kernel benchmark."""
    data = np.random.randn(2000, 500)
    return _time_fn(
        lambda: np.argsort(data, axis=1, kind="stable"), n_runs=3
    )


def bench_argsort_quick():
    """np.argsort (quicksort) on 2000x500 — potential rank speedup."""
    data = np.random.randn(2000, 500)
    return _time_fn(
        lambda: np.argsort(data, axis=1, kind="quicksort"), n_runs=3
    )


def bench_matmul():
    """np.dot on 500x500 matrices — BLAS benchmark."""
    a = np.random.randn(500, 500)
    b = np.random.randn(500, 500)
    return _time_fn(lambda: np.dot(a, b))


def bench_linregress():
    """scipy.stats.linregress on 252-point series x 100 assets."""
    from scipy.stats import linregress
    x = np.arange(252, dtype=np.float64)
    data = np.random.randn(252, 100)
    def run():
        for col in range(100):
            linregress(x, data[:, col])
    return _time_fn(run, n_runs=3)


def bench_import_zipline():
    """Time to import zipline (cold import simulation)."""
    import importlib
    # Can't truly measure cold import in-process, but we can time
    # the module-level attribute access as a proxy.
    t0 = time.perf_counter()
    from zipline import run_algorithm  # noqa: F401
    from zipline.api import order, symbol, record  # noqa: F401
    from zipline.pipeline import Pipeline  # noqa: F401
    from zipline.pipeline.factors import Returns, SimpleMovingAverage  # noqa: F401
    t1 = time.perf_counter()
    return {
        "median_ms": (t1 - t0) * 1000,
        "min_ms": (t1 - t0) * 1000,
        "max_ms": (t1 - t0) * 1000,
        "runs": 1,
        "note": "single run (import is cached after first)",
    }


# ---------------------------------------------------------------------------
# Suite definition
# ---------------------------------------------------------------------------

BENCHMARKS = {
    # Core pipeline operations
    "rank_ordinal_500x100": bench_rank_ordinal_small,
    "rank_ordinal_2000x500": bench_rank_ordinal_large,
    "rank_average_500x100": bench_rank_average,
    "factorize_10k": bench_factorize_strings,
    "factorize_100k": bench_factorize_strings_large,
    # Window operations
    "window_500x100_w20": bench_window_iterate,
    "window_2000x500_w20": bench_window_iterate_large,
    # Adjustments
    "adjustment_mutate_100x50": bench_adjustment_mutate,
    # Aggregations (SMA, StdDev)
    "nanmean_252x500": bench_nanmean,
    "nanstd_252x500": bench_nanstd,
    # Sort kernel (ranking bottleneck)
    "argsort_stable_2000x500": bench_argsort_stable,
    "argsort_quick_2000x500": bench_argsort_quick,
    # BLAS benchmark
    "matmul_500x500": bench_matmul,
    # Scipy regression
    "linregress_252x100": bench_linregress,
    # Import time
    "import_zipline": bench_import_zipline,
}

QUICK_BENCHMARKS = {
    "rank_ordinal_500x100",
    "rank_average_500x100",
    "factorize_10k",
    "window_500x100_w20",
    "nanmean_252x500",
    "argsort_stable_2000x500",
    "matmul_500x500",
}


def run_suite(quick=False):
    """Run benchmarks and return results dict."""
    from zipline.utils.hardware_profile import get_profile

    profile = get_profile()
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware_profile": {
            "name": profile.name,
            "arch": profile.arch,
            "system": profile.system,
            "cpu_name": profile.cpu_name,
            "physical_cores": profile.physical_cores,
            "logical_cores": profile.logical_cores,
            "ram_gb": round(profile.ram_gb, 1),
            "blas_provider": profile.blas_provider,
            "capabilities": sorted(profile.capabilities),
            "optimal_threads": profile.optimal_threads,
        },
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "benchmarks": {},
    }

    names = QUICK_BENCHMARKS if quick else set(BENCHMARKS.keys())
    ordered = [k for k in BENCHMARKS if k in names]

    for name in ordered:
        fn = BENCHMARKS[name]
        sys.stdout.write(f"  {name:.<40s} ")
        sys.stdout.flush()
        try:
            result = fn()
            results["benchmarks"][name] = result
            median = result["median_ms"]
            print(f"{median:>10.3f} ms")
        except Exception as e:
            results["benchmarks"][name] = {"error": str(e)}
            print(f"{'ERROR':>10s} ({e})")

    return results


def print_summary(results):
    """Print a formatted summary table."""
    profile = results["hardware_profile"]
    print("\n" + "=" * 65)
    print(f"  Hardware: {profile['name']} | {profile['cpu_name']}")
    print(f"  Arch: {profile['arch']} | Cores: {profile['physical_cores']}P/{profile['logical_cores']}L")
    print(f"  RAM: {profile['ram_gb']} GB | BLAS: {profile['blas_provider']}")
    print(f"  Caps: {', '.join(profile['capabilities']) or 'none'}")
    print(f"  Python {results['python_version']} | NumPy {results['numpy_version']} | Pandas {results['pandas_version']}")
    print("=" * 65)

    # Highlight optimization opportunities
    benchmarks = results["benchmarks"]
    print("\nOptimization Opportunities:")

    if "rank_average_500x100" in benchmarks and "rank_ordinal_500x100" in benchmarks:
        avg = benchmarks["rank_average_500x100"]
        ord_ = benchmarks["rank_ordinal_500x100"]
        if "median_ms" in avg and "median_ms" in ord_:
            ratio = avg["median_ms"] / max(ord_["median_ms"], 0.001)
            print(f"  - rank(average) is {ratio:.1f}x slower than rank(ordinal) "
                  f"-> Cython 'average' impl would close this gap")

    if "argsort_stable_2000x500" in benchmarks and "argsort_quick_2000x500" in benchmarks:
        stable = benchmarks["argsort_stable_2000x500"]
        quick = benchmarks["argsort_quick_2000x500"]
        if "median_ms" in stable and "median_ms" in quick:
            speedup = stable["median_ms"] / max(quick["median_ms"], 0.001)
            print(f"  - quicksort is {speedup:.2f}x vs stable sort "
                  f"-> potential rank speedup for ordinal method")

    if "matmul_500x500" in benchmarks:
        mm = benchmarks["matmul_500x500"]
        if "median_ms" in mm:
            gflops = (2 * 500**3) / (mm["median_ms"] / 1000) / 1e9
            print(f"  - BLAS matmul: {gflops:.1f} GFLOPS "
                  f"({profile['blas_provider']})")


def save_results(results):
    """Save results to benchmarks/ directory."""
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    profile_name = results["hardware_profile"]["name"]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = benchmarks_dir / f"{profile_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def compare_results(files):
    """Compare benchmark results from multiple JSON files."""
    datasets = []
    for f in files:
        with open(f) as fh:
            datasets.append(json.load(fh))

    if len(datasets) < 2:
        print("Need at least 2 files to compare.")
        return

    # Header
    print("\n" + "=" * 78)
    print("  Cross-Platform Benchmark Comparison")
    print("=" * 78)

    # Machine info
    for i, ds in enumerate(datasets):
        p = ds["hardware_profile"]
        print(f"\n  [{i+1}] {p['name']} | {p['cpu_name']} | {p['arch']}")
        print(f"      {p['physical_cores']}P/{p['logical_cores']}L cores | "
              f"{p['ram_gb']} GB RAM | BLAS={p['blas_provider']}")

    # Collect all benchmark names
    all_names = []
    for ds in datasets:
        for name in ds["benchmarks"]:
            if name not in all_names:
                all_names.append(name)

    # Table
    print("\n  " + "-" * 74)
    header = f"  {'Benchmark':<35s}"
    for i, ds in enumerate(datasets):
        label = ds["hardware_profile"]["name"][:12]
        header += f" {label:>12s}"
    if len(datasets) == 2:
        header += f" {'ratio':>8s}"
    print(header)
    print("  " + "-" * 74)

    for name in all_names:
        row = f"  {name:<35s}"
        times = []
        for ds in datasets:
            b = ds["benchmarks"].get(name, {})
            t = b.get("median_ms")
            times.append(t)
            if t is not None:
                row += f" {t:>10.3f}ms"
            else:
                row += f" {'n/a':>12s}"

        if len(datasets) == 2 and times[0] is not None and times[1] is not None:
            ratio = times[0] / max(times[1], 0.001)
            row += f" {ratio:>7.2f}x"

        print(row)

    print("  " + "-" * 74)


def main():
    parser = argparse.ArgumentParser(description="Zipline benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Run quick subset")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--save", action="store_true", help="Save to benchmarks/")
    parser.add_argument(
        "--compare", nargs="+", metavar="FILE",
        help="Compare 2+ benchmark JSON files instead of running benchmarks",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
        return

    if not args.json:
        print("Zipline Benchmark Suite")
        print("-" * 65)

    results = run_suite(quick=args.quick)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
