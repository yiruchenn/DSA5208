# Technical handoff to the PDF report authors

This is background guidance, **not a finished PDF report**. Replace placeholders with the team's own observed Atlas/Vanda results, plots, software versions, and names/IDs.

## Abstract / introduction points

We study a synchronous five-point Jacobi solver on a 2D rectangular grid, with fixed zero Dirichlet boundary conditions. It is a canonical local-neighbor iterative solver for the discrete Laplace equation and is closely related to diffusion smoothing. We compare a sequential C++ reference with an MPI domain decomposition, an MPI+OpenMP hybrid implementation, and two halo communication mechanisms. Unlike independent per-item processing, each iteration must exchange updated boundary rows with neighboring MPI ranks. No external dataset is used.

## Numerical formulation

For interior `1 <= i <= nx`, `1 <= j <= ny`:

\[
 u_{i,j}^{(k+1)} = \frac{1}{4}\left(u_{i-1,j}^{(k)} + u_{i+1,j}^{(k)} + u_{i,j-1}^{(k)} + u_{i,j+1}^{(k)}\right).
\]

All boundary cells are fixed at zero. The initial condition consists of two sine modes:

\[
 u_{i,j}^{(0)} = \sin\left(\frac{\pi i}{n_x+1}\right)\sin\left(\frac{\pi j}{n_y+1}\right)
 +0.25\sin\left(\frac{2\pi i}{n_x+1}\right)\sin\left(\frac{3\pi j}{n_y+1}\right).
\]

Each discrete sine mode is an eigenvector of the update operator, with eigenvalue

\[
 \lambda_{p,q} = \frac12 \left[\cos\left(\frac{p\pi}{n_x+1}\right) + \cos\left(\frac{q\pi}{n_y+1}\right)\right].
\]

The independently known solution at iteration K is therefore:

\[
 u_{i,j}^{(K)} = \lambda_{1,1}^{K}\sin\left(\frac{\pi i}{n_x+1}\right)\sin\left(\frac{\pi j}{n_y+1}\right)
 +0.25\lambda_{2,3}^{K}\sin\left(\frac{2\pi i}{n_x+1}\right)\sin\left(\frac{3\pi j}{n_y+1}\right).
\]

The code calculates both `max_error` and `relative_l2_error` against this formula (except that a relative error is reported as 0 when the exact norm is below 1e-280). Explain that this is the **discrete Jacobi update's exact result**, not an approximate continuous PDE solution. It is independent of any MPI implementation and is a strong numerical correctness check. For the small regression grids, also compare entire parallel solutions to serial CSVs.

## Decomposition and design choices

1. Row-wise, noncyclic decomposition: each MPI rank owns `nx/P` rows plus one extra for the first `nx%P` ranks. This avoids empty subdomains and balances work (differences <= 1 row).
2. Two ghost rows and zero left/right boundaries. Only `ny` doubles per neighbor are communicated, not full rows including padding. MPI neighbor ranks are `rank-1` and `rank+1`; physical boundaries use `MPI_PROC_NULL`.
3. Two alternate exchange mechanisms. **Blocking** first completes `MPI_Sendrecv` transfers, then updates every owned row. **Overlap** initiates `MPI_Isend/Irecv`, updates interior owned rows in parallel, waits, then updates first and last owned rows. At 1–2 rows/rank no interior update can overlap communication; highlight this as a limitation.
4. Hybrid threading: `#pragma omp parallel for schedule(runtime)` over owned rows, with options for static or dynamic scheduling. All MPI operations occur outside OpenMP regions; request `MPI_THREAD_FUNNELED`.
5. Double-buffered arrays: swap `current`/`next` after each step, preserving synchronous Jacobi semantics. This also avoids threads writing into each other's read dependencies.
6. Timing: wall-clock iteration loop only, with one barrier before the timer, then `MPI_MAX` of rank times; no initialization, diagnostics, CSV collection, or process startup. The maximum rank time represents overall job completion time.

For 1D row decomposition, each internal rank sends up to `2*ny` double values and receives up to `2*ny` values per iteration, distributed between two neighbors. Total per-rank stencil work is `Theta(nx*ny*K/P)`, exchange bandwidth costs are `Theta(ny*K)` per interior rank, plus a per-message latency cost. The computation-to-communication ratio degrades as `P` increases at fixed `nx`. With a 1D decomposition, the rank count cannot exceed `nx`, and the surface-to-volume ratio becomes unfavorable at high rank counts. An alternative 2D decomposition would reduce halo length asymptotically for very large P, but is not implemented; don't claim it is.

A simple performance model is `T ~ a*K*nx*ny/(P*Tthreads) + b*K*ny + c*K` (omits cache and topology; nonblocking communication can overlap part of the message time with interior work). Do not use the same symbol `T` for time and thread count in the report without distinguishing them.

## Expected hypotheses to evaluate (NOT measured results)

- For sufficient grid size, more cores should initially reduce wall time; strong scaling efficiency may fall due to halo exchange, thread creation/scheduling, memory bandwidth, network latency, and smaller work per rank.
- On a single node, MPI-only and hybrid MPI+OpenMP may scale differently. Threads reduce the number of MPI rank boundaries for a fixed core budget, but threading may add scheduling overhead and memory bandwidth contention.
- Nonblocking communication may be faster when enough interior rows exist to conceal communication time, but can be slower for small local domains due to request management overhead or lack of asynchronous progress.
- Static scheduling is expected to suit uniformly costly rows, while dynamic scheduling adds overhead without useful load balancing for this regular stencil.
- Do **not** assert any of these as observed without real cluster data. Overlap may not lead to acceleration depending on the MPI implementation's progress behavior.

## Experimental protocol and metrics

Strong scaling: use identical `(nx, ny, K)` for serial and all parallel configurations; vary ranks and thread counts, controlling total `P*Tthreads`. Repeat at least 3 times after a warmup and report median iteration time.

\[
 S(P,T_{\mathrm{threads}}) = \frac{t_{\mathrm{serial}}}{t_{P,T_{\mathrm{threads}}}},\qquad
 E(P,T_{\mathrm{threads}}) = \frac{S(P,T_{\mathrm{threads}})}{P T_{\mathrm{threads}}}.
\]

Weak scaling: fix approximately `(nx*ny)/(P*Tthreads)` by scaling both grid dimensions with `sqrt(P*Tthreads)`, while keeping `K` fixed. Since actual grid side lengths are rounded, record dimensions. The conventional weak scaling efficiency is

\[
 E_{\mathrm{weak}}(P,T_{\mathrm{threads}}) = \frac{t_{\mathrm{serial,base}}}{t_{\mathrm{parallel,scaled}}}.
\]

The scripts generate `results/strong_scaling.csv` and `results/weak_scaling.csv`, including repeat times, median, speedup/efficiency, mode, schedule and error. Use `scripts/plot_results.py` to make figures after running jobs. For comparability, prefer CPU nodes with consistent specs, fixed core counts, similar affinity, and low contention. Include raw logs with submission time/job ID if requested.

Suggested report table columns: `cluster`, `nodes`, `MPI ranks`, `OMP threads/rank`, `mode`, `problem`, `iteration count`, `median runtime (s)`, `speedup`, `efficiency`, `max numerical error`.

## Validation language (fill with actual test output)

- The test script generates a serial reference field and checks it against the exact Jacobi eigenmode solution for each test case.
- For all tested MPI rank/thread/mode configurations, the script gathers the full parallel field and compares each cell against the serial reference (`<2e-12` maximum difference). It separately asserts max analytic error `<1e-11` for those regression cases.
- State the test environment and whether **all** MPI tests actually passed. Do not insert unobserved test results.
- The fixed-iteration tests verify correctness independently of performance measurements.

## AI use disclosure (adapt to reality)

Example transparent statement for the report, to be edited by team members:

> AI-based assistance was used to help draft and review the C++ serial/MPI/OpenMP implementation, test scripts, README, and documentation. Team members reviewed the numerical discretization and communication design, compiled and ran the code, inspected numerical validation output, and collected cluster timing results. The conclusions and plotted results are based on our own executions rather than AI-generated measurements.

Only claim human review, executed tests, or collected timing if those steps truly occurred. Include specific tools and any additional report-writing assistance as applicable.

## Logistics and boundaries

- Project brief deadline: **13 November 2026** on Canvas.
- Submit this source package and the group's own **PDF report**, containing all names/IDs, algorithm, design, validation, scaling/efficiency, runtime figures **from Atlas or Vanda**, and AI disclosure.
- No real dataset: the brief's real-data conclusion requirement is conditional and does not apply.
- Package does **not** include fake cluster measurement CSV or fake plots.
