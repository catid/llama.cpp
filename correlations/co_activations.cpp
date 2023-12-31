#include "co_activations.hpp"

#include "mmapfile.hpp"
#include "compress.hpp"

CoactivationPredictor::~CoactivationPredictor()
{
    SIMDSafeFree(Buffer);
}

bool CoactivationPredictor::ReadFile(const std::string& file_path)
{
    // FIXME

    return true;
}
