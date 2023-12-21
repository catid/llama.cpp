#include "correlations.h"

#include <stdio.h>
#include <string.h>

#include <stdexcept>
#include <thread>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <regex>
#include <fstream>

static const float kNegativeThreshold = -0.5f;
static const float kPositiveThreshold = 0.5f;


//------------------------------------------------------------------------------
// Tools

static bool matchAndExtractNumber(const char* src0_name, int& extractedNumber)
{
    std::regex pattern("blk.([0-9]{1,2}).ffn_down.weight");
    std::smatch matches;
    std::string s = src0_name;

    if (std::regex_search(s, matches, pattern)) {
        if (matches.size() == 2) { // matches[0] is the whole string, matches[1] is the first group
            extractedNumber = std::stoi(matches[1].str());
            return true;
        }
    }

    return false;
}


//------------------------------------------------------------------------------
// Correlation Recorder

class CorrelationRecorder
{
public:
    void RecordTensor(int block_number, const struct ggml_tensor * src0);
    void WriteResults();

protected:
    struct ThreadContext
    {
        int ThreadId = -1;
        std::vector<int> Activations;
        std::vector<float> Row;
    };

    std::vector<std::shared_ptr<ThreadContext>> Contexts;

    struct BlockContext
    {
        int BlockNumber = -1;

        std::mutex HistogramLock;
        int HistogramWidth = -1;
        std::atomic<uint32_t>* Histogram = nullptr;

        ~BlockContext()
        {
            delete Histogram;
        }

        void RecordRow(ThreadContext* ctx, int batch, float* row, int count);
        void WriteHistogramToFile(const std::string& filename);
    };
    std::vector<std::shared_ptr<BlockContext>> Blocks;
};

static CorrelationRecorder m_CorrelationRecorder;

extern "C" {

void RecordCorrelations_WriteResults()
{
    m_CorrelationRecorder.WriteResults();
}

void RecordCorrelations_MulMat(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
            struct ggml_tensor * dst)
{
    int block_number = -1;
    if (!matchAndExtractNumber(src0->name, block_number)) {
        return;
    }

#if 0
    printf("ggml_cl_mul_mat src0:t=%d:%s[%d,%d,%d] x src1:t=%d:%s[%d,%d,%d] -> dst:t=%d:%s[%d,%d,%d]\n",
        (int)src0->type, src0->name, (int)src0->ne[0], (int)src0->ne[1], (int)src0->ne[2],
        (int)src1->type, src1->name, (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2],
        (int)dst->type, dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2]);
#endif

    m_CorrelationRecorder.RecordTensor(block_number, src1);
}

void RecordCorrelations_Activation(
    const struct ggml_tensor * src0,
            struct ggml_tensor * dst)
{
#if 0
    if (0 != strcmp(dst->name, "ffn_silu-28")) {
        return;
    }

#if 0
    printf("SILU src0:t=%d:%s[%d,%d,%d] -> dst:t=%d:%s[%d,%d,%d]\n",
        (int)src0->type, src0->name, (int)src0->ne[0], (int)src0->ne[1], (int)src0->ne[2],
        (int)dst->type, dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2]);
#endif

    m_CorrelationRecorder.RecordTensor(block_number, dst);
#endif
}

} // extern "C"

void CorrelationRecorder::WriteResults()
{
    for (auto& block : Blocks)
    {
        std::string filename = "correlations_block_";
        filename += std::to_string(block->BlockNumber);

        block->WriteHistogramToFile(filename);
    }
}

void CorrelationRecorder::RecordTensor(int block_number, const struct ggml_tensor * tensor)
{
    if (block_number < 0 || block_number > 99) {
        printf("Ignoring unusual block_number=%d\n", block_number);
        return;
    }

    while ((int)Blocks.size() <= block_number) {
        auto block = std::make_shared<BlockContext>();
        block->BlockNumber = (int)Blocks.size();
        Blocks.push_back(block);
    }
    auto& block = Blocks[block_number];

    int nthread = std::thread::hardware_concurrency() * 2;

    int n_batch = tensor->ne[1];
    int batch_per_thread = (n_batch + nthread - 1) / nthread;

    std::vector<std::thread> workers;

    Contexts.resize(nthread);
    for (int i = 0; i < nthread; ++i) {
        if (!Contexts[i]) {
            Contexts[i] = std::make_shared<ThreadContext>();
            Contexts[i]->ThreadId = i;
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

    for (int thr_index = 0; thr_index < nthread; ++thr_index)
    {
        int thr_start = thr_index * batch_per_thread;
        int thr_count = batch_per_thread;
        if (thr_start + thr_count > n_batch) {
            thr_count = n_batch - thr_start;
        }
        if (thr_count <= 0) {
            break;
        }
        ThreadContext* ctx = Contexts[thr_index].get();

        auto fn = [this, tensor, qtype, &block](ThreadContext* ctx, int start, int count)
        {
            int nelements = tensor->ne[0];
            if ((int)ctx->Row.size() != nelements) {
                ctx->Row.resize(nelements);
            }
            float* out_row = ctx->Row.data();

            for (int i = 0; i < count; ++i) {
                int batch_index = start + i;
                uint8_t* in_row = (uint8_t*)tensor->data + tensor->nb[1] * batch_index;

                if (tensor->type == GGML_TYPE_F32) {
                    out_row = (float*)in_row;
                } else if (tensor->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((ggml_fp16_t *)in_row, out_row, nelements);
                } else {
                    qtype.to_float(in_row, out_row, nelements);
                }

                block->RecordRow(ctx, (int)batch_index, out_row, (int)nelements);
            }
        };

        workers.emplace_back(fn, ctx, thr_start, thr_count);
    }

    for (auto & w : workers) { w.join(); }
}

void CorrelationRecorder::BlockContext::RecordRow(ThreadContext* ctx, int batch, float* row, int count)
{
    auto& activations = ctx->Activations;
    activations.clear();

    for (int i = 0; i < count; ++i) {
        float value = row[i];
        if (value > kNegativeThreshold && value < kPositiveThreshold) {
            continue;
        }

        activations.push_back(i);
    }

    if (activations.size() > 6000) {
        printf("WARNING: Correlating large number of activations=%d\n", (int)activations.size());
    }

    {
        std::lock_guard<std::mutex> locker(HistogramLock);
        if (!Histogram) {
            HistogramWidth = count;
            const int elements = count * (count + 1) / 2;
            Histogram = new std::atomic<uint32_t>[elements];
        }
    }

    const int size = (int)activations.size();

    for (int i = 0; i < size; ++i) {
        const int ai = activations[i];
        // Note: activations[] list is sorted.
        // First  row (ai=0) has 1 element  (offset = 0).
        // Second row (ai=1) has 2 elements (offset = 1).
        // Third  row (ai=2) has 3 elements (offset = 3).
        // Fourth row (ai=3) has 4 elements (offset = 6).
        int offset = ai * (ai + 1) / 2; // Sum of all the rows before

        for (int j = 0; j < i; ++j) {
            int index = offset + activations[j];

            ++Histogram[index];
        }
    }
}

void CorrelationRecorder::BlockContext::WriteHistogramToFile(const std::string& filename)
{
    uint64_t t0 = ggml_time_us();

    std::lock_guard<std::mutex> locker(HistogramLock);

    if (!Histogram) {
        printf("No data to write\n");
        return;
    }

    // Open a file in binary mode
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    const uint32_t width = static_cast<uint32_t>( HistogramWidth );
    file.write(reinterpret_cast<const char*>(&width), sizeof(width));

    // Write the elements of the array
    const int elements = static_cast<int>( HistogramWidth * (HistogramWidth + 1) / 2 );
    for (int i = 0; i < elements; ++i) {
        const uint32_t value = Histogram[i].load(); // Convert std::atomic to uint32_t
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    uint64_t t1 = ggml_time_us();
    printf("Wrote %d element triangular correlation matrix in %f msec\n", elements, (t1 - t0)/1000.f);
}
