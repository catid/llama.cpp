#include "correlations.h"

#include <stdio.h>
#include <stdexcept>
#include <thread>
#include <string.h>


//------------------------------------------------------------------------------
// Correlation Recorder

#include <memory>
#include <vector>
#include <atomic>

class CorrelationRecorder
{
public:
    CorrelationRecorder();
    ~CorrelationRecorder();

    void Record(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst);
    void Record_SILU(
        const struct ggml_tensor * src0,
              struct ggml_tensor * dst);

    static CorrelationRecorder* GetInstance();

protected:
    struct ThreadContext
    {
        std::vector<float> Row;
    };

    std::vector<std::shared_ptr<ThreadContext>> Contexts;

    std::atomic<uint32_t>* Histogram = nullptr;

    void RecordRow(int batch, float* row, int count);
};

CorrelationRecorder::CorrelationRecorder()
{
    Histogram = new std::atomic<uint32_t>[11008 * 11008];
}

CorrelationRecorder::~CorrelationRecorder()
{
    for (int i = 0; i < 11008; ++i) {
        printf("%d ", (int)Histogram[i]); 
    }

    delete Histogram;
}

static CorrelationRecorder m_CorrelationRecorder;

extern "C" {

void RecordCorrelations(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
            struct ggml_tensor * dst)
{
    m_CorrelationRecorder.Record(src0, src1, dst);
}

void RecordCorrelations_SILU(
    const struct ggml_tensor * src0,
            struct ggml_tensor * dst)
{
    m_CorrelationRecorder.Record_SILU(src0, dst);
}

} // extern "C"

void CorrelationRecorder::Record_SILU(
        const struct ggml_tensor * src0,
              struct ggml_tensor * dst)
{
    if (0 != strcmp(dst->name, "ffn_silu-28")) {
        return;
    }

#if 0
    printf("SILU src0:t=%d:%s[%d,%d,%d] -> dst:t=%d:%s[%d,%d,%d]\n",
        (int)src0->type, src0->name, (int)src0->ne[0], (int)src0->ne[1], (int)src0->ne[2],
        (int)dst->type, dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2]);
#endif

    const struct ggml_tensor* tensor = dst;

    if (tensor->type == GGML_TYPE_F32) {
        size_t n_batch = tensor->ne[1];
        for (size_t i = 0; i < n_batch; ++i) {
            uint8_t* row_data = (uint8_t*)tensor->data + tensor->nb[1] * i;
            float* row = (float*)row_data;
            RecordRow((int)i, row, (int)tensor->ne[0]);
        }
        return;
    }
}

void CorrelationRecorder::Record(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst)
{
    if (0 != strcmp(src0->name, "blk.30.ffn_down.weight")) {
        return;
    }

#if 0
    printf("ggml_cl_mul_mat src0:t=%d:%s[%d,%d,%d] x src1:t=%d:%s[%d,%d,%d] -> dst:t=%d:%s[%d,%d,%d]\n",
        (int)src0->type, src0->name, (int)src0->ne[0], (int)src0->ne[1], (int)src0->ne[2],
        (int)src1->type, src1->name, (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2],
        (int)dst->type, dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2]);
#endif

    return;

    const struct ggml_tensor* tensor = src1;

    if (tensor->type == GGML_TYPE_F32) {
        size_t n_batch = tensor->ne[1];
        for (size_t i = 0; i < n_batch; ++i) {
            uint8_t* row_data = (uint8_t*)tensor->data + tensor->nb[1] * i;
            float* row = (float*)row_data;
            RecordRow((int)i, row, (int)tensor->ne[0]);
        }
        return;
    }

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error("tensor type does not support to_float");
        }
    } else if (tensor->type != GGML_TYPE_F16) {
        printf("tensor->type = %d\n", (int)tensor->type);
        throw std::runtime_error("unsupported tensor type - requires FP16 or quantized");
    }

    size_t nthread = std::thread::hardware_concurrency();
    size_t n_batch = tensor->ne[1];
    size_t batch_per_thread = (n_batch + nthread - 1) / nthread;

    std::vector<std::thread> workers;

    Contexts.resize(nthread);
    for (size_t i = 0; i < nthread; ++i) {
        if (!Contexts[i]) {
            Contexts[i] = std::make_shared<ThreadContext>();
        }
    }

    for (size_t thr_index = 0; thr_index < nthread; ++thr_index)
    {
        size_t thr_start = thr_index * batch_per_thread;
        size_t thr_count = batch_per_thread;
        if (thr_start + thr_count > n_batch) {
            thr_count = n_batch - thr_start;
        }
        ThreadContext* ctx = Contexts[thr_index].get();

        auto fn = [this, tensor, qtype](ThreadContext* ctx, size_t start, size_t count)
        {
            size_t nelements = tensor->ne[0];
            if (ctx->Row.size() != nelements) {
                ctx->Row.resize(nelements);
            }
            float* out_row = ctx->Row.data();

            for (size_t i = 0; i < count; ++i) {
                size_t batch_index = start + i;
                uint8_t* in_row = (uint8_t*)tensor->data + tensor->nb[1] * batch_index;

                if (tensor->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((ggml_fp16_t *)in_row, out_row, nelements);
                } else {
                    qtype.to_float(in_row, out_row, nelements);
                }

                RecordRow((int)batch_index, out_row, (int)nelements);
            }
        };

        workers.emplace_back(fn, ctx, thr_start, thr_count);
    }

    for (auto & w : workers) { w.join(); }
}

void CorrelationRecorder::RecordRow(int batch, float* row, int count)
{
    std::vector<int> activations;

    for (int i = 0; i < count; ++i) {
        float value = row[i];
        if (value < 0.5f) {
            continue;
        }

        activations.push_back(i);
    }

    for (int i : activations)
    {
        for (int j : activations)
        {
            ++Histogram[i * 11008 + j];
        }
    }
}
