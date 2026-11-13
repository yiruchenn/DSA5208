# Project 3 — Hybrid MPI + OpenMP 2D Jacobi Heat Solver

**Language:** C++17 | **Parallel frameworks:** MPI and OpenMP | **Algorithm:** 2D, five-point Jacobi iteration with zero Dirichlet boundaries.

This code package accompanies the team's PDF report. The program implements **the same numerical algorithm** in serial and in multiple parallel configurations and supports correctness checks, strong/weak scaling, and comparisons of blocking versus nonblocking halo communication and static versus dynamic OpenMP scheduling.

## 1. Requirements

- C++17 compiler, e.g. `g++` (GCC recommended)
- MPI compiler wrapper and runtime, e.g. `mpicxx` + `mpiexec` (Open MPI or MPICH)
- OpenMP support in the MPI wrapper's compiler (GCC flag `-fopenmp`)
- Python 3.8+ (standard library only for the benchmark and validation scripts)
- Optional: `matplotlib` to create report figures

On managed clusters, load the appropriate compiler/MPI modules first, e.g. using `module avail`, followed by the exact versions available on the system. Do **not** assume the module names are identical across Atlas/Vanda or other clusters.

## 2. Build

```bash
make
```

Produces:

- `build/heat_serial` — independent, single-threaded CPU reference implementation.
- `build/heat_parallel` — MPI distributed-memory + OpenMP shared-memory implementation.

The Makefile supports changing compilers, e.g. `make MPICXX=mpic++ CXX=g++`. Clean with `make clean`.

## 3. Quick start

```bash
# Serial baseline, fixed 256 x 256 interior and 200 Jacobi steps
./build/heat_serial --nx 256 --ny 256 --steps 200

# MPI only: four processes, one thread per process
OMP_NUM_THREADS=1 mpiexec -n 4 ./build/heat_parallel \
    --nx 256 --ny 256 --steps 200 --mode blocking

# Hybrid: two MPI processes, two OpenMP threads each
OMP_NUM_THREADS=2 mpiexec -n 2 ./build/heat_parallel \
    --nx 256 --ny 256 --steps 200 --mode overlap

# Experiment with dynamic scheduling (usually higher overhead for this regular grid)
OMP_NUM_THREADS=2 mpiexec -n 2 ./build/heat_parallel \
    --nx 256 --ny 256 --steps 200 --mode overlap --schedule dynamic --chunk 8
```

For Open MPI running as root in a disposable local container, root-specific flags may be required; **do not** use those flags on shared cluster login nodes.

### Command-line options

| Option | Meaning | Default |
|---|---|---|
| `--nx N` | Number of interior rows in the entire 2D grid | 256 |
| `--ny N` | Number of interior columns | 256 |
| `--steps K` | Fixed number of Jacobi iterations | 200 |
| `--mode blocking\|overlap` | MPI halo strategy (`heat_parallel` only) | blocking |
| `--schedule static\|dynamic` | OpenMP row schedule (`heat_parallel` only) | static |
| `--chunk C` | OpenMP schedule chunk | 1 |
| `--dump FILE.csv` | Export all interior cells (for diagnostics; not timed) | disabled |
| `--help` | Show help | — |

Requirements: `nx, ny >= 2`, `steps >= 0`, and `MPI ranks <= nx`. All runs use the same deterministic initial condition and boundary values. The serial binary accepts `--mode`/`--schedule` for CLI symmetry, but does not use MPI/OpenMP.

### Machine-readable results

Each successful execution writes exactly one line starting with `RESULT` (on MPI rank 0). For example, the fields include:

```text
RESULT kind=parallel nx=... ny=... steps=... ranks=... threads=... mode=... schedule=... chunk=... seconds=... sum=... l2=... weighted=... max_error=... abs_l2_error=... relative_l2_error=... mpi_wait_seconds=... compute_seconds=...
```

`seconds` is the maximum, across ranks, of time spent in the **iteration loop only**, measured after an initial `MPI_Barrier`. It excludes allocation, initialization, analytic validation, gathering, file I/O, and startup/shutdown. This is the timing used in the benchmark; it is not the full end-to-end application time.

`mpi_wait_seconds` is blocking halo exchange elapsed time in `blocking` mode, versus waiting in `MPI_Waitall` in `overlap` mode; posted nonblocking calls are not included in the latter. Thus, it is **not** a fully comparable measure of all MPI overhead; interpret the **total** `seconds` for performance comparisons.

## 4. Numerical algorithm and parallelization

Each iteration updates the interior unknowns of a rectangular grid using the *old* values of their four neighbors:

```text
u_new(i,j) = 0.25 * (u_old(i-1,j) + u_old(i+1,j)
                  + u_old(i,j-1) + u_old(i,j+1))
```

This is the classical five-point Jacobi method for the discrete steady-state Laplace equation (or a simple diffusion/smoothing process). Boundary values on all four edges stay at zero. **Double buffering** ensures updates never use partially updated values from the same iteration.

The initial field combines two sine eigenmodes:

```text
u0(i,j) = sin(pi*i/(nx+1))*sin(pi*j/(ny+1))
        + 0.25*sin(2*pi*i/(nx+1))*sin(3*pi*j/(ny+1))
```

This choice allows independent validation after any number of Jacobi iterations. For mode `(p,q)`, its exact per-step amplification factor is

```text
lambda(p,q) = [cos(p*pi/(nx+1)) + cos(q*pi/(ny+1))]/2
```

The field after K iterations equals the above two initial modes multiplied by `lambda(1,1)^K` and `lambda(2,3)^K`. See `docs/REPORT_HANDOFF.md` for the mathematical description in LaTeX.

### MPI row decomposition

MPI ranks each own contiguous horizontal strips (balanced within one row) and **two ghost/halo rows**. They exchange their first/last owned rows with adjacent ranks every iteration. Each rank owns entire rows, so it never needs left/right MPI exchanges. The global top/bottom and all left/right physical boundaries remain zero. The computational work is `O(nx*ny*steps)` and the halo traffic is `O(ny*steps)` per interior rank (two neighboring row messages per step). The algorithm **is not embarrassingly parallel**: each time step has true neighbor dependencies.

### MPI communication strategies

- `blocking`: two `MPI_Sendrecv` exchanges finish before computing any rows.
- `overlap`: initiate `MPI_Irecv`/`MPI_Isend`, compute rows `2..local_rows-1` using OpenMP while messages are in flight, wait on all requests, then compute boundary-owned rows. Works even with one or two owned rows (where overlap is limited).

MPI is called only from the main thread (`MPI_THREAD_FUNNELED`). All interior work is parallelized across OpenMP threads using `schedule(runtime)` with static/dynamic settings.

## 5. Correctness and validation

**Recommended:** run after compiling (on a cluster, use a compute-node allocation rather than the login node):

```bash
make test
# or: python3 scripts/validate.py
```

The test script checks multiple nonsquare and very small grids, zero iterations, several MPI rank counts, different OpenMP thread counts, both communication modes, and dynamic scheduling. It:

1. Checks the serial solution against the **analytic eigenmode solution**.
2. Checks every parallel cell against the serial reference using separate exported CSV fields (not just a checksum).
3. Checks each MPI run against the **analytic solution** independently.
4. Aborts with nonzero exit status on a mismatch. Tolerances: `2e-12` for individual parallel-versus-serial cells, `1e-11` for max analytic error (these are sensible for the included small-grid tests).

If you have no MPI runtime on your *local* laptop, `make build/heat_serial && python3 scripts/validate.py --serial-only` validates just the serial algorithm. **That is not a substitute for running the full MPI tests on the cluster.**

For manual visual inspection, compare `--dump baseline.csv` and `--dump parallel.csv`. Do not use `--dump` when collecting performance times.

## 6. Benchmarks and report figures

### Strong scaling (fixed total problem size)

```bash
python3 scripts/benchmark.py --scaling strong --nx 1024 --ny 1024 \
    --steps 400 --ranks 1,2,4 --threads 1,2 --max-cores 8 \
    --repeats 3 --warmup 1 --output results/strong_scaling.csv
```

For `P` MPI processes, `T` threads each, and `T_serial` baseline runtime:

- Speedup: `S(P,T) = T_serial / T_parallel(P,T)`.
- Efficiency: `E(P,T) = S(P,T) / (P*T)`.

The CSV stores every timed repeat, its median, processor counts, schedule/mode, error, hardware hostname, and scheduler job ID when available. The serial baseline is measured with the same grid and step count. **Keep the total allocated CPU cores fixed within your resource limit; do not oversubscribe.** Do not include MPI startup / file output in timing.

### Weak scaling (approximately fixed work per core)

```bash
python3 scripts/benchmark.py --scaling weak --nx 512 --ny 512 \
    --steps 300 --ranks 1,2,4 --threads 1,2 --max-cores 8 \
    --repeats 3 --warmup 1 --output results/weak_scaling.csv
```

`--nx`/`--ny` represent the **one-worker** base size; grid dimensions grow by approximately `sqrt(P*T)` per dimension. Thus the total number of interior cells grows roughly with the worker count (rounding is inevitable). The `efficiency` column is `T_serial(base problem)/T_parallel(scaled problem)`, the conventional weak-scaling efficiency. The `speedup` field in a weak-scaling CSV is merely the same runtime ratio and should **not** be called a strong-scaling speedup.

### Charts

Run on a system with matplotlib (plotting is **not** required on the compute nodes):

```bash
python3 -m pip install matplotlib
python3 scripts/plot_results.py results/strong_scaling.csv
python3 scripts/plot_results.py results/weak_scaling.csv
```

Produces runtime, speedup, and efficiency PNG plots under `results/`. Use **actual Atlas/Vanda measurements** in the report; this repository includes no invented cluster performance data.

To compare scheduling, add `--schedules static,dynamic --chunk 8` (be aware it increases the experiment count). For clean comparisons, request the same core allocation, use the same binary and grid sizes, warm up, repeat, and report the median.

## 7. Atlas/Vanda or other HPC clusters

Two *examples*, because systems do not necessarily have the same scheduler or compiler modules:

- PBS Pro: `qsub cluster/run_pbs.pbs` (edit resource/project/queue requests as needed).
- Slurm: `sbatch cluster/run_slurm.sbatch` (edit partition/account requests as needed).

**Check the actual site's documentation first.** NUS publicly documents PBS Pro, but the project brief itself does not specify the scheduler or available modules for Atlas/Vanda. If a cluster uses another scheduler, run the same `make`, `scripts/validate.py`, and `scripts/benchmark.py` commands inside its allocation. On managed clusters, compile and launch under the MPI environment provided by that cluster, and ensure affinity matches your CPU allocation.

Store the emitted CSV files and scheduler job logs; the report requires genuine runtimes from Atlas or Vanda. Also record the exact cluster, compiler/MPI versions, number of nodes, CPU model, resource allocation, and timing protocol in the report.

## 8. File inventory

| Path | Purpose |
|---|---|
| `src/common.hpp` | Shared CLI/parser, deterministic initial field, analytic formula, error metrics, output, CSV dump. |
| `src/serial.cpp` | Single-thread Jacobi reference (double-buffered grid). |
| `src/parallel.cpp` | Hybrid MPI+OpenMP Jacobi, domain decomposition, blocking/nonblocking halos. |
| `Makefile` | Build commands for serial and MPI binaries; `make test`. |
| `scripts/validate.py` | Automated analytic and cell-by-cell regression tests. |
| `scripts/benchmark.py` | Warmed, repeated strong/weak scaling experiments; CSV. |
| `scripts/plot_results.py` | PNG runtime/speedup/efficiency plots from actual CSV. |
| `cluster/run_pbs.pbs` | Example PBS batch script; resource requests may require editing. |
| `cluster/run_slurm.sbatch` | Example Slurm batch script; alternative only. |
| `docs/REPORT_HANDOFF.md` | Formulas, design decisions, measurement guidance, and AI usage disclosure points for report writers. |
| `docs/FILE_GUIDE_CN.md` | Brief Chinese code/file walkthrough for team handoff. |
| `docs/LOCAL_TEST_STATUS.md` | Transparent log of what was and was not verified locally. |
| `results/` | Location for experimental data and figures; none fabricated or pre-populated. |

## 9. AI use

AI assistance was used to help draft/structure parts of this code package and its documentation. The team must **disclose this in the PDF report** as required by the instructor, describe human review and actual test evidence, and adapt the description to their real workflow. See `docs/REPORT_HANDOFF.md`. Do not claim MPI tests or Atlas/Vanda experiments ran unless you have actually run them.
