#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace heat {
constexpr double PI = 3.141592653589793238462643383279502884;

struct Options {
    int nx = 256;
    int ny = 256;
    int steps = 200;
    std::string mode = "blocking";   // only used by MPI version
    std::string schedule = "static"; // only used by MPI/OpenMP version
    int chunk = 1;
    std::string dump;                // optionally export full solution to CSV
};

inline int parse_nonnegative_int(const std::string& value, const std::string& name) {
    std::size_t pos = 0;
    long long parsed = 0;
    try { parsed = std::stoll(value, &pos); }
    catch (const std::exception&) { throw std::invalid_argument("Invalid integer for " + name + ": " + value); }
    if (pos != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("Out-of-range integer for " + name + ": " + value);
    return static_cast<int>(parsed);
}

inline void usage(std::ostream& out, const char* program) {
    out << "Usage: " << program << " [--nx N] [--ny N] [--steps K] "
           "[--mode blocking|overlap] [--schedule static|dynamic] "
           "[--chunk N] [--dump FILE.csv]\n"
           "Default: --nx 256 --ny 256 --steps 200 --mode blocking "
           "--schedule static --chunk 1\n";
}

inline Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(std::cout, argv[0]); std::exit(0); }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--nx") o.nx = parse_nonnegative_int(value, arg);
        else if (arg == "--ny") o.ny = parse_nonnegative_int(value, arg);
        else if (arg == "--steps") o.steps = parse_nonnegative_int(value, arg);
        else if (arg == "--mode") o.mode = value;
        else if (arg == "--schedule") o.schedule = value;
        else if (arg == "--chunk") o.chunk = parse_nonnegative_int(value, arg);
        else if (arg == "--dump") o.dump = value;
        else throw std::invalid_argument("Unknown option: " + arg);
    }
    if (o.nx < 2 || o.ny < 2) throw std::invalid_argument("--nx and --ny must be >= 2");
    if (o.chunk < 1) throw std::invalid_argument("--chunk must be >= 1");
    if (o.mode != "blocking" && o.mode != "overlap") throw std::invalid_argument("Unknown --mode");
    if (o.schedule != "static" && o.schedule != "dynamic") throw std::invalid_argument("Unknown --schedule");
    if (o.nx > std::numeric_limits<int>::max() - 2 ||
        o.ny > std::numeric_limits<int>::max() - 2)
        throw std::invalid_argument("Grid dimensions are too large for integer indexing");
    const std::size_t nrows = static_cast<std::size_t>(o.nx) + 2;
    const std::size_t ncols = static_cast<std::size_t>(o.ny) + 2;
    if (nrows > std::numeric_limits<std::size_t>::max() / ncols ||
        nrows * ncols > std::vector<double>().max_size())
        throw std::invalid_argument("Grid dimensions are too large");
    return o;
}

inline std::size_t index(int row, int col, int stride) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride)
         + static_cast<std::size_t>(col);
}

inline double mode_value(int i, int j, int nx, int ny, int p, int q) {
    return std::sin((PI * p * i) / (nx + 1.0)) *
           std::sin((PI * q * j) / (ny + 1.0));
}

inline double initial_value(int i, int j, int nx, int ny) {
    return mode_value(i, j, nx, ny, 1, 1)
         + 0.25 * mode_value(i, j, nx, ny, 2, 3);
}

inline double eigenvalue(int nx, int ny, int p, int q) {
    return 0.5 * (std::cos(PI * p / (nx + 1.0)) +
                  std::cos(PI * q / (ny + 1.0)));
}

inline double exact_value(int i, int j, int nx, int ny, int steps) {
    return std::pow(eigenvalue(nx, ny, 1, 1), steps) * mode_value(i, j, nx, ny, 1, 1)
         + 0.25 * std::pow(eigenvalue(nx, ny, 2, 3), steps) * mode_value(i, j, nx, ny, 2, 3);
}

struct Stats {
    double sum = 0;
    double sum_sq = 0;
    double weighted_sum = 0;
    double error_sq = 0;
    double exact_sq = 0;
    double max_error = 0;
};

inline Stats local_stats(const std::vector<double>& grid, int local_rows,
                         int global_first_row, int nx, int ny, int steps) {
    Stats s;
    const int stride = ny + 2;
    for (int li = 1; li <= local_rows; ++li) {
        const int gi = global_first_row + li - 1;
        for (int j = 1; j <= ny; ++j) {
            const double u = grid[index(li, j, stride)];
            const double exact = exact_value(gi, j, nx, ny, steps);
            const double err = u - exact;
            s.sum += u;
            s.sum_sq += u * u;
            s.weighted_sum += u * (1.0 + 0.001 * gi + 0.00001 * j);
            s.error_sq += err * err;
            s.exact_sq += exact * exact;
            s.max_error = std::max(s.max_error, std::abs(err));
        }
    }
    return s;
}

inline void print_result(const Options& o, const std::string& kind,
                         int ranks, int threads, const std::string& mode,
                         double seconds, const Stats& s,
                         double wait_seconds = 0.0, double compute_seconds = 0.0) {
    // One machine-readable record for scripts/benchmark.py and scripts/validate.py.
    const double abs_l2 = std::sqrt(s.error_sq);
    const double exact_l2 = std::sqrt(s.exact_sq);
    const double relative = exact_l2 > 1e-280 ? abs_l2 / exact_l2 : 0.0;
    std::cout << std::setprecision(17)
              << "RESULT kind=" << kind
              << " nx=" << o.nx << " ny=" << o.ny << " steps=" << o.steps
              << " ranks=" << ranks << " threads=" << threads
              << " mode=" << mode << " schedule=" << o.schedule
              << " chunk=" << o.chunk << " seconds=" << seconds
              << " sum=" << s.sum << " l2=" << std::sqrt(s.sum_sq)
              << " weighted=" << s.weighted_sum
              << " max_error=" << s.max_error
              << " abs_l2_error=" << abs_l2
              << " relative_l2_error=" << relative
              << " mpi_wait_seconds=" << wait_seconds
              << " compute_seconds=" << compute_seconds << '\n';
}

inline void write_full_csv(const std::string& path, const std::vector<double>& dense,
                           int nx, int ny) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out << "i,j,value\n" << std::setprecision(17);
    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            out << i << ',' << j << ',' << dense[static_cast<std::size_t>(i - 1) * ny + (j - 1)] << '\n';
    if (!out) throw std::runtime_error("Failed writing " + path);
}

} // namespace heat
