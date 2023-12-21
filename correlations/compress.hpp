#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <functional>

static const uint32_t kCorrelationFileHead = 0xca7dca7d;

bool WriteCorrelationMatrix(
    const uint32_t* matrix_data,
    int matrix_width,
    int block_number,
    const std::string& file_path);

class CorrelationMatrix
{
public:
    ~CorrelationMatrix();

    int BlockNumber = -1;
    int MatrixWidth = -1;
    int WordCount = -1;
    uint32_t* Data = nullptr;

    bool ReadFile(const std::string& file_path);
    void Accumulate(const CorrelationMatrix& other);
    bool WriteFile(const std::string& file_path);
};

// Write out some randomized data and read it back in to verify the code works
bool CorrelationMatrix_UnitTest();


//------------------------------------------------------------------------------
// Tools

uint8_t* SIMDSafeAllocate(size_t size);
void SIMDSafeFree(void* ptr);

struct ScopedF
{
    ScopedF(std::function<void()> func) {
        F = func;
    }
    ~ScopedF() {
        F();
    }
    std::function<void()> F;
};
