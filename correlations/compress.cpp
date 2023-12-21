#include "compress.hpp"


//------------------------------------------------------------------------------
// Zstd

#include <zstd.h> // Zstd

void ZstdCompress(
    const std::vector<uint8_t>& uncompressed,
    std::vector<uint8_t>& compressed)
{
    compressed.resize(ZSTD_compressBound(uncompressed.size()));
    const size_t size = ZSTD_compress(
        compressed.data(),
        compressed.size(),
        uncompressed.data(),
        uncompressed.size(),
        kZstdLevel);
    if (ZSTD_isError(size)) {
        compressed.clear();
        return;
    }
    compressed.resize(size);
}

bool ZstdDecompress(
    const uint8_t* compressed_data,
    int compressed_bytes,
    int uncompressed_bytes,
    std::vector<uint8_t>& uncompressed)
{
    uncompressed.resize(uncompressed_bytes);
    const size_t size = ZSTD_decompress(
        uncompressed.data(),
        uncompressed.size(),
        compressed_data,
        compressed_bytes);
    if (ZSTD_isError(size)) {
        return false;
    }
    if (size != static_cast<size_t>( uncompressed_bytes )) {
        return false;
    }
    return true;
}


//------------------------------------------------------------------------------
// CorrelationMatrix

bool WriteCorrelationMatrix(
    std::atomic<uint32_t>* matrix_data,
    int matrix_width,
    int block_number,
    const std::string& file_path)
{

}

bool CorrelationMatrix::ReadFile(const std::string& file_path)
{

}

bool CorrelationMatrix::Accumulate(const CorrelationMatrix& other)
{
    const uint32_t* src = other.Data;
    uint32_t* dst = Data;

    const int words = WordCount;
    for (int i = 0; i < words; ++i) {
        dst[i] += src[i];
    }
}
