#!/usr/bin/env python3
"""Run reproducible strong/weak scaling benchmarks; record wall times to CSV.

Usage: python3 scripts/benchmark.py --scaling strong --ranks 1,2,4,8 --threads 1,2
"""
import argparse
import csv
import datetime
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["timestamp_utc", "hostname", "job_id", "scaling", "kind", "nx", "ny",
          "steps", "ranks", "threads", "total_cores", "mode", "schedule", "chunk",
          "repeats", "times_seconds", "median_seconds", "serial_baseline_seconds",
          "speedup", "efficiency", "max_error", "relative_l2_error"]


def result_line(command, env):
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"Exit {proc.returncode}: {' '.join(command)}\n"
                           f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    result = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")]
    if len(result) != 1:
        raise RuntimeError(f"Missing/duplicate RESULT records from {' '.join(command)}\n{proc.stdout}")
    return dict(token.split("=", 1) for token in result[0].split()[1:])


def sample(command, env, repeats, warmup):
    for _ in range(warmup):
        result_line(command, env)
    records = [result_line(command, env) for _ in range(repeats)]
    times = [float(row["seconds"]) for row in records]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise RuntimeError(f"Invalid timing data for {' '.join(command)}: {times}")
    if any(float(x["max_error"]) > 1e-10 for x in records):
        raise RuntimeError(f"Analytic correctness check failed: {records}")
    return statistics.median(times), times, max(float(x["max_error"]) for x in records), \
           max(float(x["relative_l2_error"]) for x in records)


def row_from_results(args, *, kind, nx, ny, ranks, threads, mode, schedule, chunk,
                     measured, times, max_error, relative_error, serial_baseline):
    workers = ranks * threads
    # Strong scaling uses the same domain and serial runtime for all workers.
    # Weak scaling uses a fixed per-worker area, compared with one-worker serial runtime.
    speedup = serial_baseline / measured
    efficiency = speedup / workers if args.scaling == "strong" else speedup
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hostname=platform.node(), job_id=os.getenv("PBS_JOBID", os.getenv("SLURM_JOB_ID", "")),
                scaling=args.scaling, kind=kind, nx=nx, ny=ny, steps=args.steps,
                ranks=ranks, threads=threads, total_cores=workers, mode=mode,
                schedule=schedule, chunk=chunk, repeats=args.repeats,
                times_seconds=";".join(f"{t:.9g}" for t in times),
                median_seconds=f"{measured:.9g}",
                serial_baseline_seconds=f"{serial_baseline:.9g}",
                speedup=f"{speedup:.9g}", efficiency=f"{efficiency:.9g}",
                max_error=f"{max_error:.9g}", relative_l2_error=f"{relative_error:.9g}")


def parse_positive_csv(value):
    items = [int(x.strip()) for x in value.split(",")]
    if not items or min(items) < 1: raise argparse.ArgumentTypeError("Use comma-separated positive integers")
    return list(dict.fromkeys(items))


def parse_names(value, valid):
    items = [s.strip() for s in value.split(",")]
    if not items or any(s not in valid for s in items):
        raise argparse.ArgumentTypeError(f"Must be a comma-separated subset of {sorted(valid)}")
    return list(dict.fromkeys(items))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scaling", choices=("strong", "weak"), default="strong")
    p.add_argument("--nx", type=int, default=512, help="Strong: global nx; weak: one-worker base nx")
    p.add_argument("--ny", type=int, default=512, help="Strong: global ny; weak: one-worker base ny")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--ranks", type=parse_positive_csv, default=[1, 2, 4])
    p.add_argument("--threads", type=parse_positive_csv, default=[1, 2])
    p.add_argument("--modes", type=lambda v: parse_names(v, {"blocking", "overlap"}),
                   default=["blocking", "overlap"])
    p.add_argument("--schedules", type=lambda v: parse_names(v, {"static", "dynamic"}),
                   default=["static"])
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--max-cores", type=int, default=8,
                   help="Skip configurations with ranks*threads exceeding this allocation")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--mpiexec", default=os.getenv("MPIEXEC", "mpiexec"))
    p.add_argument("--output", default=None, help="Default: results/<scaling>_scaling.csv")
    args = p.parse_args()
    if any(v < 1 for v in (args.nx, args.ny, args.steps, args.chunk,
                           args.max_cores, args.repeats)) or args.warmup < 0:
        p.error("nx, ny, steps, chunk, max-cores and repeats must be positive; warmup >= 0")
    if args.nx < 2 or args.ny < 2: p.error("nx, ny must be >= 2")
    if not shutil.which(args.mpiexec): p.error(f"MPI runtime not found: {args.mpiexec}")
    serial_exe = ROOT / "build/heat_serial"
    parallel_exe = ROOT / "build/heat_parallel"
    if not serial_exe.is_file() or not parallel_exe.is_file(): p.error("Run `make` first")
    configurations = [(r, t) for r in args.ranks for t in args.threads
                      if r * t <= args.max_cores]
    if not configurations: p.error("No rank/thread configuration fits --max-cores")
    if args.scaling == "strong" and any(r > args.nx for r, _ in configurations):
        p.error("For strong scaling, ranks must not exceed nx")

    baseline_env = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    base_command = [str(serial_exe), "--nx", str(args.nx), "--ny", str(args.ny),
                    "--steps", str(args.steps)]
    serial_time, serial_times, serial_err, serial_rel = sample(
        base_command, baseline_env, args.repeats, args.warmup)
    print(f"serial ({args.nx}x{args.ny}): median {serial_time:.6g}s", flush=True)
    rows = [row_from_results(args, kind="serial", nx=args.nx, ny=args.ny,
                             ranks=1, threads=1, mode="none", schedule="static", chunk=1,
                             measured=serial_time, times=serial_times, max_error=serial_err,
                             relative_error=serial_rel, serial_baseline=serial_time)]
    for ranks, threads in configurations:
        nworkers = ranks * threads
        if args.scaling == "strong":
            nx, ny = args.nx, args.ny
        else:
            factor = math.sqrt(nworkers)
            nx = max(2, round(args.nx * factor))
            ny = max(2, round(args.ny * factor))
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                   OMP_PROC_BIND="close", OMP_PLACES="cores")
        for mode in args.modes:
            for schedule in args.schedules:
                command = [args.mpiexec, "-n", str(ranks), str(parallel_exe),
                           "--nx", str(nx), "--ny", str(ny), "--steps", str(args.steps),
                           "--mode", mode, "--schedule", schedule,
                           "--chunk", str(args.chunk)]
                measured, timings, err, rel = sample(command, env, args.repeats, args.warmup)
                row = row_from_results(args, kind="parallel", nx=nx, ny=ny,
                                       ranks=ranks, threads=threads, mode=mode,
                                       schedule=schedule, chunk=args.chunk,
                                       measured=measured, times=timings,
                                       max_error=err, relative_error=rel,
                                       serial_baseline=serial_time)
                rows.append(row)
                print(f"{ranks} MPI x {threads} OpenMP ({mode}/{schedule}, {nx}x{ny}): "
                      f"{measured:.6g}s, efficiency={float(row['efficiency']):.3f}", flush=True)
    output = ROOT / (args.output or f"results/{args.scaling}_scaling.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
