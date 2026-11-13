#include "common.hpp"
#include <mpi.h>
#include <omp.h>
#include <array>
#include <climits>

// 1D row decomposition: each rank owns whole contiguous rows plus two ghost rows.
// Exactly one thread (the master) calls MPI: MPI_THREAD_FUNNELED.
int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << "ERROR: MPI does not support MPI_THREAD_FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    try {
        const heat::Options o = heat::parse_options(argc, argv);
        if (nranks > o.nx) throw std::invalid_argument("MPI ranks must be <= --nx (at least one owned row per rank)");
        if (!o.dump.empty() && static_cast<long long>(o.nx) * o.ny > INT_MAX)
            throw std::invalid_argument("--dump needs nx*ny <= INT_MAX for MPI_Gatherv");

        omp_set_dynamic(0);
        omp_set_schedule(o.schedule == "static" ? omp_sched_static : omp_sched_dynamic, o.chunk);
        const int threads = omp_get_max_threads();
        const int local_rows = o.nx / nranks + (rank < o.nx % nranks ? 1 : 0);
        const int first_row = rank * (o.nx / nranks) + std::min(rank, o.nx % nranks) + 1;
        const int stride = o.ny + 2;
        const std::size_t cells = static_cast<std::size_t>(local_rows + 2) * stride;
        std::vector<double> current(cells, 0.0), next(cells, 0.0);
        for (int li = 1; li <= local_rows; ++li)
            for (int j = 1; j <= o.ny; ++j)
                current[heat::index(li, j, stride)] =
                    heat::initial_value(first_row + li - 1, j, o.nx, o.ny);

        const int above = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int below = rank == nranks - 1 ? MPI_PROC_NULL : rank + 1;
        // Use only interior columns in messages; 0 and ny+1 remain zero.
        const int width = o.ny;
        double total_wait = 0.0, total_compute = 0.0;
        auto compute_rows = [&](int begin, int end) {
            if (begin > end) return;
            const double t0 = MPI_Wtime();
            #pragma omp parallel for schedule(runtime)
            for (int i = begin; i <= end; ++i) {
                for (int j = 1; j <= o.ny; ++j) {
                    next[heat::index(i, j, stride)] = 0.25 * (
                        current[heat::index(i - 1, j, stride)] +
                        current[heat::index(i + 1, j, stride)] +
                        current[heat::index(i, j - 1, stride)] +
                        current[heat::index(i, j + 1, stride)]);
                }
            }
            total_compute += MPI_Wtime() - t0;
        };

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        for (int step = 0; step < o.steps; ++step) {
            if (o.mode == "blocking") {
                const double t0 = MPI_Wtime();
                // Tags distinguish upward (101) and downward (102) row traffic.
                MPI_Sendrecv(&current[heat::index(1, 1, stride)], width, MPI_DOUBLE,
                             above, 101, &current[heat::index(local_rows + 1, 1, stride)],
                             width, MPI_DOUBLE, below, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&current[heat::index(local_rows, 1, stride)], width, MPI_DOUBLE,
                             below, 102, &current[heat::index(0, 1, stride)],
                             width, MPI_DOUBLE, above, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_wait += MPI_Wtime() - t0;
                compute_rows(1, local_rows);
            } else {
                // Post transfers, compute rows that DO NOT depend on incoming halos.
                std::array<MPI_Request, 4> requests{};
                MPI_Irecv(&current[heat::index(0, 1, stride)], width, MPI_DOUBLE,
                          above, 102, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(&current[heat::index(local_rows + 1, 1, stride)], width,
                          MPI_DOUBLE, below, 101, MPI_COMM_WORLD, &requests[1]);
                MPI_Isend(&current[heat::index(1, 1, stride)], width, MPI_DOUBLE,
                          above, 101, MPI_COMM_WORLD, &requests[2]);
                MPI_Isend(&current[heat::index(local_rows, 1, stride)], width,
                          MPI_DOUBLE, below, 102, MPI_COMM_WORLD, &requests[3]);
                compute_rows(2, local_rows - 1);
                const double t0 = MPI_Wtime();
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                total_wait += MPI_Wtime() - t0;
                compute_rows(1, 1);
                if (local_rows > 1) compute_rows(local_rows, local_rows);
            }
            current.swap(next);
        }
        const double elapsed = MPI_Wtime() - start;
        double max_elapsed = 0.0, max_wait = 0.0, max_compute = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_wait, &max_wait, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        const heat::Stats local = heat::local_stats(current, local_rows, first_row,
                                                    o.nx, o.ny, o.steps);
        std::array<double, 5> partial = {
            local.sum, local.sum_sq, local.weighted_sum, local.error_sq, local.exact_sq
        };
        std::array<double, 5> reduced{};
        MPI_Reduce(partial.data(), reduced.data(), static_cast<int>(partial.size()),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double max_error = 0.0;
        MPI_Reduce(&local.max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            heat::Stats stats;
            stats.sum = reduced[0]; stats.sum_sq = reduced[1];
            stats.weighted_sum = reduced[2]; stats.error_sq = reduced[3];
            stats.exact_sq = reduced[4]; stats.max_error = max_error;
            heat::print_result(o, "parallel", nranks, threads, o.mode,
                               max_elapsed, stats, max_wait, max_compute);
        }

        if (!o.dump.empty()) {
            // Diagnostics excluded from the measured iteration time.
            std::vector<double> packed(static_cast<std::size_t>(local_rows) * o.ny);
            for (int li = 1; li <= local_rows; ++li)
                for (int j = 1; j <= o.ny; ++j)
                    packed[static_cast<std::size_t>(li - 1) * o.ny + j - 1] =
                        current[heat::index(li, j, stride)];
            std::vector<int> counts, displacements;
            std::vector<double> dense;
            if (rank == 0) {
                counts.resize(nranks);
                displacements.resize(nranks);
                dense.resize(static_cast<std::size_t>(o.nx) * o.ny);
                for (int r = 0; r < nranks; ++r) {
                    const int rows = o.nx / nranks + (r < o.nx % nranks ? 1 : 0);
                    const int first = r * (o.nx / nranks) + std::min(r, o.nx % nranks);
                    counts[r] = rows * o.ny;
                    displacements[r] = first * o.ny;
                }
            }
            MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                        rank == 0 ? dense.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0) heat::write_full_csv(o.dump, dense, o.nx, o.ny);
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR (rank " << rank << "): " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
