# Local verification status (2026-10-30)

This record is included for transparent handoff. It is **not** an Atlas/Vanda benchmark result.

## Executed with the available development tools

- Compiled `src/serial.cpp` with GNU `g++`, `-std=c++17 -O3 -Wall -Wextra -Wpedantic` successfully.
- Executed `python3 scripts/validate.py --serial-only`: all five analytic-reference cases passed, with max error up to approximately `4.72e-15` in those cases.
- Compiled `src/parallel.cpp` with GNU `g++ -fopenmp` against a **temporary simulated MPI API** (located outside this package) and ran the solver with 1/2/3/4 simulated ranks, multiple OpenMP threads, both communication modes, and static/dynamic schedules. Every simulated field-wise comparison matched serial for the tested cases.
- Ran the strong/weak benchmark orchestration and plotting scripts using simulated MPI outputs; verified CSV and PNG generation. **The resulting simulated timings were deleted and must not be used in the report.**
- Python scripts passed `python3 -m py_compile`.

## Not yet verified in this development environment

The environment used here does **not** provide `mpicxx`, an actual MPI runtime, or Atlas/Vanda allocation. Consequently:

- The distributed executable has **not** been built or executed against an actual MPI implementation here.
- The PBS/Slurm job scripts have **not** been submitted on Atlas/Vanda.
- No genuine cluster runtimes, scaling numbers, or scheduler logs are included.

**Before submission, the team must** load the cluster's MPI toolchain, run `make`, run `python3 scripts/validate.py` inside a suitable allocation, and then run `scripts/benchmark.py` with real MPI ranks. If the cluster uses a scheduler with different resources/module syntax, edit the example job script accordingly. Use **only these real CSV/log outputs** in the report's runtime figures and scaling claims.
