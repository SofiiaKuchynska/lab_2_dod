// Variant: 11 (exclusive_scan)
// Standard: C++20
// Compiler: MinGW-w64 GCC 13+ (CLion Bundled MinGW, Ninja)
// Input file format:
// Line 1: N (number of elements)
// Lines 2..N+1: integer values (one per line)

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <execution>
#include <thread>
#include <barrier>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

template <typename Func>
double time_it(Func&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    return DurationMs(t1 - t0).count();
}

void make_dataset(const std::string& path, size_t n) {
    std::ofstream out(path);
    if (!out.is_open()) return;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(-1000, 1000);

    out << n << '\n';
    for (size_t i = 0; i < n; ++i) {
        out << dist(rng) << '\n';
    }
}

std::vector<int64_t> read_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return {};

    size_t n = 0;
    in >> n;
    std::vector<int64_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        in >> data[i];
    }
    return data;
}

template <typename T, typename BinaryOp = std::plus<T>>
void custom_exclusive_scan_k(
    const std::vector<T>& in,
    std::vector<T>& out,
    T init,
    size_t K,
    BinaryOp op = std::plus<T>{}
) {
    const size_t n = in.size();
    if (n == 0) return;
    if (out.size() < n) out.resize(n);

    if (K <= 1) {
        std::exclusive_scan(in.begin(), in.end(), out.begin(), init, op);
        return;
    }

    K = std::min(K, n);

    std::vector<size_t> pos(K + 1);
    size_t base = n / K;
    size_t rem = n % K;
    pos[0] = 0;
    for (size_t i = 0; i < K; ++i) {
        pos[i + 1] = pos[i] + base + (i < rem ? 1 : 0);
    }

    std::vector<T> block_sums(K);
    std::vector<T> block_offsets(K);

    auto on_phase1_complete = [&]() noexcept {
        std::exclusive_scan(block_sums.begin(), block_sums.end(), block_offsets.begin(), init, op);
    };

    std::barrier sync_point(static_cast<ptrdiff_t>(K), on_phase1_complete);
    std::vector<std::jthread> workers;
    workers.reserve(K);

    for (size_t i = 0; i < K; ++i) {
        workers.emplace_back([&, i]() {
            size_t s = pos[i];
            size_t e = pos[i + 1];

            if (s < e) {
                std::exclusive_scan(in.begin() + s, in.begin() + e, out.begin() + s, T{}, op);
                block_sums[i] = op(out[e - 1], in[e - 1]);
            } else {
                block_sums[i] = T{};
            }

            sync_point.arrive_and_wait();

            if (s < e) {
                T offset = block_offsets[i];
                std::transform(
                    out.begin() + s,
                    out.begin() + e,
                    out.begin() + s,
                    [&](const T& val) { return op(offset, val); }
                );
            }
        });
    }
}

struct HeavyOp {
    int64_t operator()(int64_t a, int64_t b) const {
        int64_t res = a + b;
        for (int i = 0; i < 20; ++i) {
            res = (res * 6364136223846793005ULL + 1) ^ (a + i);
        }
        return res;
    }
};

template <typename Op>
void run_suite(const std::string& op_name, const std::vector<size_t>& sizes, Op binary_op) {
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t reps = 3;

    std::cout << "\nOperation: " << op_name << " (hardware threads: " << hw << ")\n";

    for (size_t n : sizes) {
        std::string fname = "data_" + std::to_string(n) + ".txt";
        if (!fs::exists(fname)) {
            make_dataset(fname, n);
        }

        std::vector<int64_t> in = read_dataset(fname);
        std::vector<int64_t> out(n);

        std::cout << "\nSize N = " << n << "\n";

        double t_def = 1e9;
        for (size_t r = 0; r < reps; ++r) {
            t_def = std::min(t_def, time_it([&]() {
                std::exclusive_scan(in.begin(), in.end(), out.begin(), 0LL, binary_op);
            }));
        }
        std::cout << "std::exclusive_scan: " << std::fixed << std::setprecision(3) << t_def << " ms\n";

        auto test_policy = [&](const char* name, auto pol) {
            double t = 1e9;
            for (size_t r = 0; r < reps; ++r) {
                t = std::min(t, time_it([&]() {
                    std::exclusive_scan(pol, in.begin(), in.end(), out.begin(), 0LL, binary_op);
                }));
            }
            std::cout << name << ": " << std::fixed << std::setprecision(3) << t << " ms\n";
        };

        test_policy("policy seq", std::execution::seq);
        test_policy("policy par", std::execution::par);
        test_policy("policy par_unseq", std::execution::par_unseq);
        test_policy("policy unseq", std::execution::unseq);

        std::cout << "\nK\tTime(ms)\tSpeedup\tK/HW\n";

        size_t max_K = std::max<size_t>(16, hw * 2);
        size_t best_k = 1;
        double best_t = 1e9;

        for (size_t k = 1; k <= max_K; ++k) {
            double t = 1e9;
            for (size_t r = 0; r < reps; ++r) {
                t = std::min(t, time_it([&]() {
                    custom_exclusive_scan_k(in, out, 0LL, k, binary_op);
                }));
            }

            if (t < best_t) {
                best_t = t;
                best_k = k;
            }

            double speedup = t_def / t;
            double ratio = static_cast<double>(k) / hw;

            std::cout << k << "\t"
                      << std::fixed << std::setprecision(3) << t << "\t\t"
                      << std::fixed << std::setprecision(2) << speedup << "x\t"
                      << std::fixed << std::setprecision(2) << ratio << "\n";
        }

        std::cout << "Optimal K = " << best_k
                  << " (time: " << best_t << " ms, ratio K/HW = "
                  << static_cast<double>(best_k) / hw << ")\n";
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<size_t> test_sizes = { 1'000'000, 5'000'000, 15'000'000 };

    run_suite("std::plus", test_sizes, std::plus<int64_t>{});
    run_suite("heavy_op", test_sizes, HeavyOp{});

    return 0;
}