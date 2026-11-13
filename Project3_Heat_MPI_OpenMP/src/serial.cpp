#include "common.hpp"
#include <chrono>

int main(int argc, char** argv) {
    try {
        const heat::Options o = heat::parse_options(argc, argv);
        const int stride = o.ny + 2;
        const std::size_t cells = static_cast<std::size_t>(o.nx + 2) * stride;
        std::vector<double> current(cells, 0.0), next(cells, 0.0);
        for (int i = 1; i <= o.nx; ++i)
            for (int j = 1; j <= o.ny; ++j)
                current[heat::index(i, j, stride)] = heat::initial_value(i, j, o.nx, o.ny);

        const auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < o.steps; ++step) {
            for (int i = 1; i <= o.nx; ++i)
                for (int j = 1; j <= o.ny; ++j)
                    next[heat::index(i, j, stride)] = 0.25 * (
                        current[heat::index(i - 1, j, stride)] +
                        current[heat::index(i + 1, j, stride)] +
                        current[heat::index(i, j - 1, stride)] +
                        current[heat::index(i, j + 1, stride)]);
            current.swap(next);  // double buffering avoids update-order errors
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const auto stats = heat::local_stats(current, o.nx, 1, o.nx, o.ny, o.steps);
        heat::print_result(o, "serial", 1, 1, "none", seconds, stats);

        if (!o.dump.empty()) {
            std::vector<double> dense(static_cast<std::size_t>(o.nx) * o.ny);
            for (int i = 1; i <= o.nx; ++i)
                for (int j = 1; j <= o.ny; ++j)
                    dense[static_cast<std::size_t>(i - 1) * o.ny + (j - 1)] =
                        current[heat::index(i, j, stride)];
            heat::write_full_csv(o.dump, dense, o.nx, o.ny);
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
