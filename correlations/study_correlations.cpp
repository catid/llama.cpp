#include "compress.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;


static uint64_t SplitMix64(uint64_t& seed)
{
    uint64_t z = (seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}


class Correlation
{
public:
    ~Correlation();

    void Calculate(CorrelationMatrix& m);

    int Width = -1;

    // Triangular matrix just like CorrelationMatrix (except this is actual correlation).
    // Lower triangle + diagonal, row-first in memory.
    float* RMatrix = nullptr;
    double LargestR = 0.0;

    // List of auto-encoder classified neurons.
    // This will also contain neurons that never were observed firing.
    std::vector<int> AutoEncoderNeurons;

    int AutoEncoderNeuronCount = 0;
    int KnowledgeNeuronCount = 0;

    // Maximum seen histogram count for knowledge-classified neurons
    uint32_t MaxKnowledgeHistValue = 0;

    double Get(int i/*row*/, int j/*column*/)
    {
        // Ensure j <= i to avoid reading outside the lower triangle.
        if (j > i) {
            std::swap(j, i);
        }

        int row_offset = i * (i + 1) / 2;
        return RMatrix[row_offset + j];
    }

    std::vector<int> RemapIndices;

protected:
    int WordCount = -1;

    // Note: The matrix is symmetric, so row stats are the same as column stats
    std::vector<double> StdDevs, Means;
};

Correlation::~Correlation()
{
    SIMDSafeFree(RMatrix);
}

void Correlation::Calculate(CorrelationMatrix& m)
{
    const int width = m.MatrixWidth;
    Width = width;

    if (WordCount != m.WordCount) {
        SIMDSafeFree(RMatrix);
        RMatrix = (float*)SIMDSafeAllocate(m.WordCount * sizeof(float));

        StdDevs.resize(width);
        Means.resize(width);
    }

    // Initialize RemapIndices
    RemapIndices.resize(width);
    for (int i = 0; i < width; ++i) {
        RemapIndices[i] = i;
    }

    AutoEncoderNeurons.clear();

    // Iterative algorithm to separate knowledge from auto-encoder neurons:

    KnowledgeNeuronCount = width;
    for (int i = 0; i < width; ++i) {
        RemapIndices[i] = i;
    }

    for (int epoch = 0; epoch < 2; ++ epoch)
    {
        // Correlation matrix calculation

        for (int i = 0; i < KnowledgeNeuronCount; ++i) {
            const int row_i = RemapIndices[i];
            // Calculate mean of row:

            uint64_t sum_values = 0;

            for (int j = 0; j < KnowledgeNeuronCount; ++j) {
                const int col_j = RemapIndices[j];

                sum_values += m.Get(row_i, col_j);
            }

            double mean = sum_values / (double)KnowledgeNeuronCount;
            Means[row_i] = mean;

            // Calculate standard deviation of row:

            double sum_sd = 0.0;

            // For each row:
            for (int j = 0; j < KnowledgeNeuronCount; ++j) {
                const int col_j = RemapIndices[j];

                uint32_t value = m.Get(row_i, col_j);
                double diff = value - mean;

                sum_sd += diff * diff;
            }

            StdDevs[row_i] = std::sqrt(sum_sd / width);
        }

        if (epoch >= 1) {
            break;
        }

        uint32_t max_value = 0;

        // Each epoch we identify the ones that are most likely
        // auto-encoder neurons, and move them to the end of the list,
        // then do it again with the remaining neurons.
        for (int i = 0; i < KnowledgeNeuronCount; ++i)
        {
            int positive = 0;

            const int row_i = RemapIndices[i];
            const int mean_i = (int)Means[row_i];

            for (int j = 0; j < KnowledgeNeuronCount; ++j) {
                if (i == j) {
                    // Indices are unique so this is an equivalent faster test
                    continue;
                }

                const int col_j = RemapIndices[j];
                const int mean_j = (int)Means[col_j];
                int value = m.Get(row_i, col_j);

                if (value < mean_i && value < mean_j) {
                    positive++;
                } else {
                    positive--;
                }
            }

            if (positive < 0) {
                AutoEncoderNeurons.push_back(row_i);
            } else {
                uint32_t value = m.Get(row_i, row_i);
                if (max_value < value) {
                    max_value = value;
                }
            }
        }

        // Sort auto-encoder neurons by how often they fire (just for fun)
        std::sort(AutoEncoderNeurons.begin(), AutoEncoderNeurons.end(), [&](int i, int j) {
            uint32_t hist_i = m.Get(i, i);
            uint32_t hist_j = m.Get(j, j);
            return hist_i > hist_j;
        });

        MaxKnowledgeHistValue = max_value;

        // Reset the map
        for (int i = 0; i < width; ++i) {
            RemapIndices[i] = i;
        }

        // Move all selected to the end (in reverse order)
        int count = 0;
        for (int i : AutoEncoderNeurons) {
            ++count;
            std::swap(RemapIndices[width - count], RemapIndices[i]);
        }

        // Sort auto-encoder neurons by how often they fire (just for fun)
        std::sort(RemapIndices.begin(), RemapIndices.end(), [&](int i, int j) {
            uint32_t hist_i = m.Get(i, i);
            uint32_t hist_j = m.Get(j, j);
            return hist_i > hist_j;
        });

        AutoEncoderNeuronCount = count;
        KnowledgeNeuronCount = width - AutoEncoderNeuronCount;

        cout << "Epoch " << epoch << ":" << endl;
        cout << "Max knowledge neuron histogram value: " << MaxKnowledgeHistValue << endl;
        cout << "Auto-Encoder neurons(" << AutoEncoderNeuronCount << "/" << width
            << ", " << (AutoEncoderNeuronCount * 100.f / width) << "%) identified" << endl;
    }

    // HACK
    for (int i = 0; i < width; ++i)
    {
        int offset = i * (i + 1) / 2;

        double norm_factor = 1.0 / m.Get(i, i);

        for (int j = 0; j <= i; ++j)
        {
            double r = m.Get(i, j) * norm_factor;

            RMatrix[offset + j] = (float)r;
        }
    }

    return;

    // Calculate r values:

    double largest_r = 0.0;
    for (int i = 0; i < KnowledgeNeuronCount; ++i)
    {
        int offset = i * (i + 1) / 2;

        const int row_i = RemapIndices[i];
        for (int j = 0; j <= i; ++j)
        {
            const int col_j = RemapIndices[j];

            double denom = StdDevs[row_i] * StdDevs[col_j];

            // Filter out cases that have 0 stddev (all constant)
            float fr = 0.0;
            if (denom != 0.0) {
                double mean_i = Means[row_i];
                double mean_j = Means[col_j];

                // Calculate covariance between row i and row j
                double sum_cov = 0.0;
                for (int k = 0; k < KnowledgeNeuronCount; ++k) {
                    const int col_k = RemapIndices[k];

                    sum_cov += (m.Get(row_i, col_k) - mean_i) * (m.Get(col_j, col_k) - mean_j);
                }

                double cov_ij = sum_cov / width;

                double r = cov_ij / denom;
                if (largest_r < r) {
                    largest_r = r;
                }

                //cout << "(" << i << ", " << j << ") = " << r << " cov_ij=" << cov_ij << " stdi=" << StdDevs[i] << " stdj=" << StdDevs[j] << " avgi=" << mean_i << " avgj=" << mean_j << endl;

                fr = (float)r;
            }

            if (!std::isfinite(fr) || fr > 1.001 || fr < -1.001) {
                cout << "BAD CORRELATION: (" << i << ", " << j << ") = " << fr << endl;
                cout << "StdDevs[i] = " << StdDevs[i] << endl;
                cout << "StdDevs[j] = " << StdDevs[j] << endl;
                cout << "Means[i] = " << Means[i] << endl;
                cout << "Means[j] = " << Means[j] << endl;
                continue;
            }

            RMatrix[offset + j] = fr;
        }
    }
    LargestR = largest_r;
}

struct SAParams
{
    int max_negative_dist = 32;
    int log2_max_move = 8;
    int max_epochs = 1000;
};

static double ScoreOrder(int knowledge_count, std::vector<int>& Indices, Correlation& corr, const SAParams& params)
{
    double score = 0.0;

    // For just everything under the diagonal:
    for (int i = 0; i < knowledge_count; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            // Get correlation matrix entry for transformed indices,
            // which does not change the correlation matrix values but just re-orders them.
            // These are guaranteed to be unique.
            const int row_i = Indices[i];
            const int col_j = Indices[j];

            double r = corr.Get(row_i, col_j);

            // distance from diagonal
            int d = i - j;
#if 0
            // Only negative correlations close to the column should be penalized
            if (r < 0.0 && d > params.max_negative_dist) {
                continue;
            }
#endif
            score += r / d;
        }
    }

    return score;
}

/*
    We want negative correlated values to be far away.
    Past a certain distance we don't care since we won't evaluate them together as neighbors.

    We want positive correlated values to be mostly equal on each side if possible to cause them to cluster.
    We don't care how far away they are, so that even far values move together.
*/
static double GetRightScore(int knowledge_count, int i, std::vector<int>& Indices, Correlation& corr, const SAParams& params)
{
    const int row_i = Indices[i];

    double score = 0.0;

    for (int j = 0; j < knowledge_count; ++j) {
        const int col_j = Indices[j];

        if (row_i == col_j) {
            continue;
        }

        const double r = corr.Get(row_i, col_j);
#if 0
        // Only negative correlations close to the column should be penalized
        if (r < 0.0 && std::abs(i - j) > params.max_negative_dist) {
            continue;
        }
#endif
        if (j > i) {
            score += r;
        } else {
            score -= r;
        }
    }

    return score;
}

static void MoveIndex(int knowledge_count, std::vector<int>& Indices, int i, int move)
{
    if (move > 0) {
        int end = i + move;
        if (end >= knowledge_count) {
            end = knowledge_count - 1;
        }

        const int t = Indices[i];
        for (int j = i; j < end; ++j) {
            Indices[j] = Indices[j + 1];
        }
        Indices[end] = t;
    } else {
        int end = i + move;
        if (end < 0) {
            end = 0;
        }

        const int t = Indices[i];
        for (int j = i; j > end; --j) {
            Indices[j] = Indices[j - 1];
        }
        Indices[end] = t;
    }
}

#include <random>
#include <algorithm>

static void SimulatedAnnealing(Correlation& corr, std::vector<int>& Indices, const SAParams& params)
{
    const int knowledge_count = corr.KnowledgeNeuronCount;

    // Shuffle indices to a random initial position
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(Indices.begin(), Indices.begin() + knowledge_count, g);

    uint64_t seed = g();

    // For each epoch:
    for (int epoch = 0; epoch < params.max_epochs; ++epoch)
    {
        const double temperature = 1.0 - (epoch + 1) / (double)params.max_epochs;

        // For each index:
        for (int i = 0; i < knowledge_count; ++i)
        {
            // If we should move left:
            double right_score = GetRightScore(knowledge_count, i, Indices, corr, params);
            if (right_score == 0.0) {
                continue; // No reason to move!
            }

            // Uniformly pick a random move magnitude
            const uint32_t move_mask = (1 << params.log2_max_move) - 1;
            int move = (uint32_t)SplitMix64(seed) & move_mask;

            // Modulate movement by temperature, reducing it slowly towards the end
            move *= temperature;
            if (move < 1) {
                move = 1;
            }

            // If we should be moving to the left:
            if (right_score < 0.0) {
                move = -move;
            }

            // Make the move
            MoveIndex(knowledge_count, Indices, i, move);
        }

        double score = ScoreOrder(knowledge_count, Indices, corr, params);
        cout << "Epoch " << epoch << " score=" << score << endl;
    }
}

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>

struct ElementInfo
{
    int Index = -1;
    double Value = 0.0;

    bool operator>(const ElementInfo& rhs) const {
        return Value > rhs.Value;
    }
};

static std::vector<int> RCMOrder(Correlation& corr)
{
    std::vector<int> indices = corr.RemapIndices;
    const int knowledge_count = corr.KnowledgeNeuronCount;

    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS, 
        property<vertex_color_t, default_color_type,
        property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    Graph G;

    std::vector<int> top_indices;

    const int k = 256;
    const int start_index = 8000;

    for (int i = 0; i < knowledge_count; ++i) {
        // Find k largest elements
        std::priority_queue<ElementInfo, std::vector<ElementInfo>, std::greater<ElementInfo>> minHeap;

        const int row_i = corr.RemapIndices[i];

        for (int j = 0; j < knowledge_count; ++j) {
            if (i == j) {
                continue;
            }

            const int col_j = corr.RemapIndices[j];

            ElementInfo info;
            info.Value = corr.Get(row_i, col_j);
            info.Index = col_j;

            if (minHeap.size() < k) {
                // If the heap is not full, add the element directly
                minHeap.push(info);
            } else if (info > minHeap.top()) {
                // If the current element is larger than the smallest in the heap,
                // replace the smallest with the current element
                minHeap.pop();
                minHeap.push(info);
            }
        }

        // Collect the K elements with largest correlation as neighbors
        std::vector<int> neighbors;
        while (!minHeap.empty()) {
            auto& top = minHeap.top();
            if (top.Value > 0.0) {
                // We only consider nodes neighbors if they have positive correlation
                neighbors.push_back(top.Index);
            }
            minHeap.pop();
        }

        for (int j : neighbors) {
            add_edge(i, j, G);
        }
    }

    std::vector<graph_traits<Graph>::vertex_descriptor> inv_perm(num_vertices(G));

    Vertex s = vertex(start_index, G);

    cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G), 
                                  get(vertex_degree, G));

    // Inverse permutation to get the RCM ordering
    std::vector<graph_traits<Graph>::vertex_descriptor> perm(num_vertices(G));
    for (std::size_t i = 0; i < inv_perm.size(); ++i)
        perm[inv_perm[i]] = i;

    // Initialize indices
    indices.resize(knowledge_count);
    for (int i = 0; i < knowledge_count; ++i) {
        indices[i] = perm[i];
    }

    SAParams params;
    double score = ScoreOrder(knowledge_count, indices, corr, params);
    cout << "Reverse Cuthill-McKee score=" << score << endl;

    return indices;
}

static void GenerateNeuronHistogram(CorrelationMatrix& m)
{
    const int width = m.MatrixWidth;

    // Histogram

    const int hist_w = 8000; int hist_h = 4000;
    uint32_t* hdata = (uint32_t*)SIMDSafeAllocate(width * 4);
    uint32_t* histogram = (uint32_t*)SIMDSafeAllocate(hist_w * 4);
    uint8_t* histogram_image = SIMDSafeAllocate(hist_w * hist_h);
    ScopedF histogram_scope([&]() {
        SIMDSafeFree(hdata);
        SIMDSafeFree(histogram);
        SIMDSafeFree(histogram_image);
    });

    for (int i = 0; i < width; ++i) {
        const int diagonal_j = i * (i + 1) / 2;
        uint32_t value = m.Data[diagonal_j];
        hdata[i] = value;
    }
    std::sort(hdata, hdata + width);

    for (int i = 0; i < hist_w; ++i) {
        histogram[i] = 0;
    }

    for (int i = 0; i < width; ++i) {
        int bin = i * hist_w / width;
        if (bin < 0) {
            bin = 0;
        } else if (bin >= width) {
            bin = width - 1;
        }

        // Note this has an aliasing artifact that will cause this to jump up and down based on variance.
        // I kind of like this because it visually demonstrates variance in the data without additional effort.
        histogram[bin] += hdata[i];
    }

    uint32_t hist_max = histogram[0];
    for (int i = 1; i < hist_w; ++i) {
        if (hist_max < histogram[i]) {
            hist_max = histogram[i];
        }
    }

    for (int i = 0; i < hist_w; ++i) {
        int height = (int)(histogram[i] / (double)hist_max * hist_h);
        if (height < 0) {
            height = 0;
        } else if (height >= hist_h) {
            height = hist_h;
        }

        for (int j = 0; j < hist_h; ++j) {
            histogram_image[i + (hist_h - 1 - j) * hist_w] = j > height ? 0 : 255;
        }
    }

    // Write histogram

    cv::Mat hist_image(hist_h, hist_w, CV_8UC1, (void*)histogram_image);

    std::string hist_filename = "histogram_block_";
    hist_filename += std::to_string(m.BlockNumber);
    hist_filename += ".png";

    cv::imwrite(hist_filename, hist_image);
}

static void GenerateHeatmap(CorrelationMatrix& m)
{
    const int width = m.MatrixWidth;

    Correlation corr;
    corr.Calculate(m);

    //std::vector<int> Indices = corr.RemapIndices;
    std::vector<int> Indices = RCMOrder(corr);

    //SAParams sa_params;
    //SimulatedAnnealing(m, corr, Indices, sa_params);

    // Generate heatmap

    uint8_t* heatmap = SIMDSafeAllocate(width * width * 3);
    ScopedF heatmap_scope([&]() {
        SIMDSafeFree(heatmap);
    });

    for (int i = 0; i < width; ++i)
    {
        // Amplify all the correlations a lot since most are pretty small
        double r_norm_factor = width / 2;

        for (int j = 0; j <= i; ++j)
        {
            const int row_i = Indices[i];
            const int row_j = Indices[j];

            double r = m.Get(row_i, row_j) * 2.0 / m.Get(row_i, row_i);
            //double r = corr.Get(row_i, row_j) * r_norm_factor;

            int heat = r * 255.0;

            int offset_ij = i * width + j;
            int offset_ji = j * width + i;
            if (heat >= 0) {
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 0] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 2] = (uint8_t)heat;
            }
            else if (heat < 0) {
                heat = -heat;
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = 0;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = 0;
                heatmap[offset_ji * 3 + 0] = 0;
                heatmap[offset_ji * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 2] = 0;
            }
        }
    }

    // Store as a PNG image

    cv::Mat heatmap_image(width, width, CV_8UC3, (void*)heatmap);

    std::string heatmap_filename = "heatmap_block_";
    heatmap_filename += std::to_string(m.BlockNumber);
    heatmap_filename += ".png";

    cv::imwrite(heatmap_filename, heatmap_image);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cerr << "Expected: study_correlations file1.zstd" << endl;
        return -1;
    }
    const char* m_file = argv[1];

    CorrelationMatrix m;

    if (!m.ReadFile(m_file)) {
        cerr << "Failed to read file: " << m_file << endl;
        return -1;
    }

    cout << "Studying " << m_file << " (Block=" << m.BlockNumber << ", Width=" << m.MatrixWidth << ")" << endl;

    GenerateNeuronHistogram(m);
    GenerateHeatmap(m);

    return 0;
}
