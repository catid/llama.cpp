#pragma once


//------------------------------------------------------------------------------
// Correlation Recorder

#ifdef ENABLE_CORRELATIONS

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

// Returns 0 on success, non-zero on test failure
int32_t RecordCorrelations_SelfTest();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENABLE_CORRELATIONS
