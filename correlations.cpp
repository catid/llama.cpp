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
#include <mutex>

class CorrelationRecorder
{
public:
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

    std::mutex HistogramLock;
    int HistogramWidth = -1;
    std::atomic<uint32_t>* Histogram = nullptr;

    void RecordRow(int batch, float* row, int count);
    void PrintHistogram(bool full = false);
};

CorrelationRecorder::~CorrelationRecorder()
{
    PrintHistogram(true);
    delete Histogram;
}

static CorrelationRecorder m_CorrelationRecorder;

extern "C" {

void RecordCorrelations(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
            struct ggml_tensor * dst)
{
    //m_CorrelationRecorder.Record(src0, src1, dst);
}

void RecordCorrelations_SILU(
    const struct ggml_tensor * src0,
            struct ggml_tensor * dst)
{
    //m_CorrelationRecorder.Record_SILU(src0, dst);
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

    const struct ggml_tensor* tensor = src1;

    //uint64_t t0 = ggml_time_us();

    size_t nthread = std::thread::hardware_concurrency() * 2;

    size_t n_batch = tensor->ne[1];
    size_t batch_per_thread = (n_batch + nthread - 1) / nthread;

    std::vector<std::thread> workers;

    Contexts.resize(nthread);
    for (size_t i = 0; i < nthread; ++i) {
        if (!Contexts[i]) {
            Contexts[i] = std::make_shared<ThreadContext>();
        }
    }

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error("tensor type does not support to_float");
        }
    } else if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
        printf("tensor->type = %d\n", (int)tensor->type);
        throw std::runtime_error("unsupported tensor type - requires FP16 or quantized");
    }

    for (size_t thr_index = 0; thr_index < nthread; ++thr_index)
    {
        size_t thr_start = thr_index * batch_per_thread;
        size_t thr_count = batch_per_thread;
        if (thr_start + thr_count > n_batch) {
            thr_count = n_batch - thr_start;
        }
        if (thr_count <= 0) {
            break;
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

                if (tensor->type == GGML_TYPE_F32) {
                    out_row = (float*)in_row;
                } else if (tensor->type == GGML_TYPE_F16) {
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

    //uint64_t t1 = ggml_time_us();
    //printf("%f msec", (t1 - t0)/100.f);

    PrintHistogram();
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

    {
        std::lock_guard<std::mutex> locker(HistogramLock);
        if (!Histogram) {
            HistogramWidth = count;
            Histogram = new std::atomic<uint32_t>[count * count];
        }
    }

    for (int i : activations)
    {
        const int offset = i * HistogramWidth;
        for (int j : activations)
        {
            ++Histogram[offset + j];
        }
    }
}

void CorrelationRecorder::PrintHistogram(bool full)
{
    if (!Histogram) {
        return;
    }

    printf("\n");
    for (int i = 0; i < HistogramWidth && i < 64; ++i) {
        printf("%d ", (int)Histogram[i]); 
    }
    printf("\n");
}
