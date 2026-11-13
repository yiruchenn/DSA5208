#!/usr/bin/env python3
"""Numerical regression tests: exact solution + field-wise serial/MPI comparison."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SERIAL = ROOT / "build/heat_serial"
PARALLEL = ROOT / "build/heat_parallel"


def run(command, env=None):
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                              text=True, check=True, timeout=120)
    except subprocess.CalledProcessError as err:
        raise AssertionError(f"Command failed: {' '.join(map(str, command))}\n"
                             f"stdout:\n{err.stdout}\nstderr:\n{err.stderr}") from err
    except subprocess.TimeoutExpired as err:
        raise AssertionError(f"Command timed out: {' '.join(map(str, command))}") from err
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if len(lines) != 1:
        raise AssertionError(f"Expected exactly one RESULT line; got:\n{proc.stdout}")
    return dict(token.split("=", 1) for token in lines[0].split()[1:])


def field(path):
    with open(path, newline="") as f:
        return [float(row["value"]) for row in csv.DictReader(f)]


def assert_match(reference, actual, description):
    assert len(reference) == len(actual), f"{description}: grid shape mismatch"
    largest = max((abs(x - y) for x, y in zip(reference, actual)), default=0.0)
    assert largest < 2e-12, f"{description}: max cell difference = {largest}"
    print(f"PASS {description}: max_cell_difference={largest:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpiexec", default=os.getenv("MPIEXEC", "mpiexec"))
    parser.add_argument("--serial-only", action="store_true",
                        help="Useful if no MPI compiler/runtime is installed")
    args = parser.parse_args()
    if not SERIAL.is_file():
        raise SystemExit("Missing serial executable: run `make` (or `make build/heat_serial`)")
    if not args.serial_only and not PARALLEL.is_file():
        raise SystemExit("Missing MPI executable: run `make` first")
    if not args.serial_only and not shutil.which(args.mpiexec):
        raise SystemExit(f"Cannot find {args.mpiexec}; install/load MPI or use --serial-only")

    cases = [(2, 3, 0), (3, 5, 1), (7, 11, 17), (17, 13, 25), (21, 29, 100)]
    with tempfile.TemporaryDirectory(prefix="heat-validation-") as tmp:
        for nx, ny, steps in cases:
            reference_path = Path(tmp) / "reference.csv"
            opts = ["--nx", str(nx), "--ny", str(ny), "--steps", str(steps)]
            result = run([str(SERIAL), *opts, "--dump", str(reference_path)])
            error = float(result["max_error"])
            assert error < 1e-11, f"serial ({nx},{ny},{steps}) error = {error}"
            print(f"PASS serial analytic ({nx}x{ny}, {steps} steps): max_error={error:.3g}")
            reference = field(reference_path)
            if args.serial_only:
                continue
            # Nonuniform decomposition and ranks with just one local row are tested.
            for ranks in sorted(set((1, min(2, nx), min(3, nx), min(4, nx)))):
                for threads in (1, 2):
                    for mode in ("blocking", "overlap"):
                        destination = Path(tmp) / "parallel.csv"
                        env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                                   OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE")
                        data = run([args.mpiexec, "-n", str(ranks), str(PARALLEL),
                                    *opts, "--mode", mode, "--dump", str(destination)], env=env)
                        assert int(data["ranks"]) == ranks
                        assert float(data["max_error"]) < 1e-11, f"MPI analytic fail: {data}"
                        assert_match(reference, field(destination),
                                     f"MPI {ranks} ranks x {threads} threads, {mode}, {nx}x{ny}, steps={steps}")
            # Additional OpenMP schedule variation.
            dest = Path(tmp) / "dynamic.csv"
            env = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE")
            run([args.mpiexec, "-n", "2", str(PARALLEL), *opts,
                 "--schedule", "dynamic", "--chunk", "2", "--dump", str(dest)], env=env)
            assert_match(reference, field(dest), f"dynamic scheduling {nx}x{ny}")
    print("ALL VALIDATION TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
