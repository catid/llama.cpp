#pragma once

#include <cstdint>
#include <string>
#include <atomic>

static const uint32_t kCorrelationFileHead = 0xca7dca7d;

bool WriteCorrelationMatrix(
    std::atomic<uint32_t>* matrix_data,
    int matrix_width,
    int block_number,
    const std::string& file_path);

class CorrelationMatrix
{
public:
    int BlockNumber = -1;
    int MatrixWidth = -1;
    int WordCount = -1;
    uint32_t* Data = nullptr;

    bool ReadFile(const std::string& file_path);
    bool Accumulate(const CorrelationMatrix& other);
};
