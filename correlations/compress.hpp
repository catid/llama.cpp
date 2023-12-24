#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <functional>

static const uint32_t kCorrelationFileHead = 0xca7dca7d;

/*
    Terminology warning:

    We are calling this object a CorrelationMatrix, however it is actually a
    histogram of the number of co-events for neurons firing.  As a result,
    this matrix is actually a joint-probability matrix.  The diagonal measures
    P(X), where X=the event neuron X fires.  P(X,Y) = P(Y,X) is the probability
    that neurons X and Y fire.  P(X|Y) = P(X,Y) / P(Y) by Bayes' theorem.

    It requires additional processing to convert this into an actual correlation
    matrix, which is still square and symmetric.  The data is easier to compress
    and download if we instead collect just the joint-probability matrix at first
    and the convert to correlation matrix after we have collected all the data
    from all the nodes in the cluster.
*/

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

    // Lower triangle + diagonal, row-first in memory.
    uint32_t* Data = nullptr;

    uint32_t Get(int i/*row*/, int j/*column*/)
    {
        // Ensure j <= i to avoid reading outside the lower triangle.
        if (j > i) {
            std::swap(j, i);
        }

        int row_offset = i * (i + 1) / 2;
        return Data[row_offset + j];
    }

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
