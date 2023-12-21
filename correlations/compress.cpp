#include "compress.hpp"
#include "mmapfile.hpp"

#include <zstd.h>

#include <cstring>
#include <iostream>
#include <vector>
using namespace std;


//------------------------------------------------------------------------------
// SIMD-Safe Aligned Memory Allocations

static const unsigned kSimdAlignmentBytes = 32;

uint8_t* SIMDSafeAllocate(size_t size)
{
    uint8_t* data = (uint8_t*)malloc(kSimdAlignmentBytes + size);
    if (!data) {
        return nullptr;
    }
    unsigned offset = (unsigned)((uintptr_t)data % kSimdAlignmentBytes);
    data += kSimdAlignmentBytes - offset;
    data[-1] = (uint8_t)offset;
    return data;
}

void SIMDSafeFree(void* ptr)
{
    if (!ptr) {
        return;
    }
    uint8_t* data = (uint8_t*)ptr;
    unsigned offset = data[-1];
    if (offset >= kSimdAlignmentBytes) {
        return;
    }
    data -= kSimdAlignmentBytes - offset;
    free(data);
}


//------------------------------------------------------------------------------
// CorrelationMatrix

bool WriteCorrelationMatrix(
    const uint32_t* matrix_data,
    int matrix_width,
    int block_number,
    const std::string& file_path)
{
    int src_bytes = 4 * matrix_width * (matrix_width + 1) / 2;

    const int compressed_max = ZSTD_compressBound(src_bytes);
    uint8_t* compressed = SIMDSafeAllocate(compressed_max);
    ScopedF compressed_scope([&]() {
        SIMDSafeFree(compressed);
    });

    const size_t compressed_size = ZSTD_compress(
        compressed,
        compressed_max,
        matrix_data,
        src_bytes,
        1/*level*/);
    if (ZSTD_isError(compressed_size)) {
        return false;
    }

    const int file_bytes = 4 + 4 + 4 + compressed_size;

    MappedFile file;
    MappedView view;

    if (!file.OpenWrite(file_path.c_str(), file_bytes)) {
        return false;
    }
    if (!view.Open(&file)) {
        return false;
    }
    if (!view.MapView(0, file.Length)) {
        return false;
    }
    if ((int)file.Length < file_bytes) {
        return false;
    }

    uint32_t* dest = reinterpret_cast<uint32_t*>(view.Data);
    dest[0] = kCorrelationFileHead;
    dest[1] = (uint32_t)block_number;
    dest[2] = (uint32_t)matrix_width;
    memcpy(dest + 3, compressed, compressed_size);

    return true;
}

CorrelationMatrix::~CorrelationMatrix()
{
    SIMDSafeFree(Data);
}

bool CorrelationMatrix::ReadFile(const std::string& file_path)
{
    MappedFile file;
    MappedView view;

    if (!file.OpenRead(file_path.c_str(), true, false)) {
        return false;
    }
    if (!view.Open(&file)) {
        return false;
    }
    if (!view.MapView(0, file.Length)) {
        return false;
    }
    if (file.Length < 12) {
        return false;
    }

    const uint32_t* src = reinterpret_cast<uint32_t*>(view.Data);
    if (src[0] != kCorrelationFileHead) {
        return false;
    }
    BlockNumber = (int)src[1];
    MatrixWidth = (int)src[2];
    WordCount = MatrixWidth * (MatrixWidth + 1) / 2;
    const int uncompressed_bytes = WordCount * 4;
    SIMDSafeFree(Data);
    Data = reinterpret_cast<uint32_t*>( SIMDSafeAllocate(uncompressed_bytes) );

    const size_t actual_bytes = ZSTD_decompress(
        Data,
        uncompressed_bytes,
        src + 3,
        file.Length - 12);
    if (ZSTD_isError(actual_bytes)) {
        return false;
    }
    if ((int)actual_bytes != uncompressed_bytes) {
        return false;
    }

    return true;
}

void CorrelationMatrix::Accumulate(const CorrelationMatrix& other)
{
    const uint32_t* src = other.Data;
    uint32_t* dst = Data;

    const int words = WordCount;
    for (int i = 0; i < words; ++i) {
        dst[i] += src[i];
    }
}

bool CorrelationMatrix::WriteFile(const std::string& file_path)
{
    if (!Data) {
        return false;
    }

    return WriteCorrelationMatrix(Data, MatrixWidth, BlockNumber, file_path);
}

bool CorrelationMatrix_UnitTest()
{
    // Generate some test data

    int matrix_width = 1337;
    int block_number = 42;
    std::string file_path = "test.zstd";

    int elements = matrix_width * (matrix_width + 1) / 2;
    std::vector<uint32_t> matrix_data(elements);

    for (int i = 0; i < elements; ++i) {
        matrix_data[i] = elements + i;
    }

    if (sizeof(std::atomic<uint32_t>) != 4) {
        throw std::runtime_error("FIXME: Need to implement intermediate conversion here on this arch");
        return false;
    }

    if (!WriteCorrelationMatrix(matrix_data.data(), matrix_width, block_number, file_path)) {
        cerr << "WriteCorrelationMatrix failed" << endl;
        return false;
    }

    CorrelationMatrix m;
    if (!m.ReadFile(file_path)) {
        cerr << "ReadFile failed" << endl;
        return false;
    }

    if (m.BlockNumber != block_number) {
        cerr << "Wrong BlockNumber" << endl;
        return false;
    }
    if (m.MatrixWidth != matrix_width) {
        cerr << "Wrong MatrixWidth" << endl;
        return false;
    }
    if (m.WordCount != elements) {
        cerr << "Wrong WordCount" << endl;
        return false;
    }
    for (int i = 0; i < elements; ++i) {
        if (m.Data[i] != elements + i) {
            cerr << "Data corrupted at " << i << endl;
            return false;
        }
    }

    cout << "CorrelationMatrix unit test passed" << endl;
    return true;
}
