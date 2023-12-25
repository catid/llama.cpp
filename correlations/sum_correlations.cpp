#include "compress.hpp"

#include <iostream>
#include <chrono>
using namespace std;

static uint64_t get_usec()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return duration.count();
}

int main(int argc, char* argv[])
{
    uint64_t t0 = get_usec();

    if (argc != 3) {
        cerr << "Usage: sum_correlations output_file.zstd input_file.zstd" << endl;
        return -1;
    }

    CorrelationMatrix m1, m2;
    const char* m1_file = argv[1];
    const char* m2_file = argv[2];

    if (!m1.ReadFile(m1_file)) {
        cerr << "Failed to open file: " << m1_file << endl;
        return -1;
    }

    if (!m2.ReadFile(m2_file)) {
        cerr << "Failed to open file: " << m2_file << endl;
        return -1;
    }

    if (!m1.Accumulate(m2)) {
        cerr << "Overflow during matrix accumulation!" << endl;
        return -1;
    }

    if (!m1.WriteFile(m1_file)) {
        cerr << "Failed to write file: " << m1_file << endl;
        return -1;
    }

    uint64_t t1 = get_usec();

    cout << "Accumulated " << m1_file << " <- " << m2_file << " in " << (t1 - t0) / 1000.f << " msec" << endl;

    return 0;
}
