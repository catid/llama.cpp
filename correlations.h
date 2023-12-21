#pragma once


//------------------------------------------------------------------------------
// Correlation Recorder

#ifdef ENABLE_CORRELATIONS

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void RecordCorrelations_MulMat(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
            struct ggml_tensor * dst);

void RecordCorrelations_Activation(
    const struct ggml_tensor * src0,
            struct ggml_tensor * dst);

void RecordCorrelations_WriteResults();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENABLE_CORRELATIONS
