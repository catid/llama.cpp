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

    uint32_t MaxValue = 0;

    double Get(int i/*row*/, int j/*column*/)
    {
        // Ensure j <= i to avoid reading outside the lower triangle.
        if (j > i) {
            int t = j;
            j = i;
            i = t;
        }

        int row_offset = i * (i + 1) / 2;
        return RMatrix[row_offset + j];
    }

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

    // Calculate max value for normalization purposes
    uint32_t max_value = 0;
    for (int i = 0; i < m.WordCount; ++i) {
        const uint32_t value = m.Data[i];
        if (max_value < value) {
            max_value = value;
        }
    }
    MaxValue = max_value;

    double norm_factor = 1.0 / max_value;

    // Correlation matrix calculation

    for (int i = 0; i < width; ++i) {
        // Calculate mean of row:

        uint64_t sum_values = 0;

        for (int j = 0; j < width; ++j) {
            sum_values += m.Get(i, j);
        }

        double mean = sum_values / (double)width * norm_factor;
        Means[i] = mean;

        // Calculate standard deviation of row:

        double sum_sd = 0.0;

        // For each row:
        for (int j = 0; j < width; ++j) {
            uint32_t value = m.Get(i, j);
            double diff = value * norm_factor - mean;
            sum_sd += diff * diff;
        }

        StdDevs[i] = std::sqrt(sum_sd);
    }

    // Calculate r values:

    double largest_r = 0.0;
    for (int i = 0; i < width; ++i)
    {
        int offset = i * (i + 1) / 2;

        for (int j = 0; j <= i; ++j)
        {
            double norm_value = m.Data[offset + j] * norm_factor;
            double cov_ij = (norm_value - Means[i]) * (norm_value - Means[j]);
            double r = cov_ij / (StdDevs[i] * StdDevs[j]);
            if (largest_r < r) {
                largest_r = r;
            }

            RMatrix[offset + j] = (float)r;
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

static double ScoreOrder(std::vector<int>& Indices, Correlation& corr, const SAParams& params)
{
    const int width = corr.Width;
    double score = 0.0;

    // For just everything under the diagonal:
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            // Get correlation matrix entry for transformed indices,
            // which does not change the correlation matrix values but just re-orders them.
            const int row_i = Indices[i];
            const int col_j = Indices[j];

            if (row_i == col_j) {
                continue;
            }

            double r = corr.Get(row_i, col_j);

            // distance from diagonal
            int d = i - j;

            // Only negative correlations close to the column should be penalized
            if (r < 0.0 && d > params.max_negative_dist) {
                continue;
            }

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
static double GetRightScore(int i, std::vector<int>& Indices, Correlation& corr, const SAParams& params)
{
    const int width = corr.Width;
    const int row_i = Indices[i];

    double score = 0.0;

    for (int j = 0; j < width; ++j) {
        const int col_j = Indices[j];

        if (row_i == col_j) {
            continue;
        }

        const double r = corr.Get(row_i, col_j);

        // Only negative correlations close to the column should be penalized
        if (r < 0.0 && std::abs(i - j) > params.max_negative_dist) {
            continue;
        }

        if (j > i) {
            score += r;
        } else {
            score -= r;
        }
    }

    return score;
}

static void MoveIndex(std::vector<int>& Indices, int i, int move)
{
    const int width = (int)Indices.size();

    if (move > 0) {
        int end = i + move;
        if (end >= width) {
            end = width - 1;
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
    const int width = corr.Width;

    // Initialize indices
    Indices.resize(width);
    for (int i = 0; i < width; ++i) {
        Indices[i] = i;
    }

    // Shuffle indices to a random initial position
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(Indices.begin(), Indices.end(), g);

    uint64_t seed = g();

    // For each epoch:
    for (int epoch = 0; epoch < params.max_epochs; ++epoch)
    {
        const double temperature = 1.0 - (epoch + 1) / (double)params.max_epochs;

        // For each index:
        for (int i = 0; i < width; ++i)
        {
            // If we should move left:
            double right_score = GetRightScore(i, Indices, corr, params);
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
            MoveIndex(Indices, i, move);
        }

        double score = ScoreOrder(Indices, corr, params);
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

static void RCMOrder(Correlation& corr, std::vector<int>& Indices)
{
    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS, 
        property<vertex_color_t, default_color_type,
        property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    Graph G;

    std::vector<int> top_indices;

    const int k = 64;
    const int start_index = 8000;

    const int width = corr.Width;
    for (int i = 0; i < width; ++i) {
        // Find k largest elements
        std::priority_queue<ElementInfo, std::vector<ElementInfo>, std::greater<ElementInfo>> minHeap;
        for (int j = 0; j < width; ++j) {
            if (i == j) {
                continue;
            }

            ElementInfo info;
            info.Value = corr.Get(i, j);
            info.Index = j;

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

    std::cout << "Reverse Cuthill-McKee ordering:" << std::endl;
    for (std::size_t i = 0; i < perm.size(); ++i)
        std::cout << perm[i] << std::endl;

    // Initialize indices
    Indices.resize(width);
    for (int i = 0; i < width; ++i) {
        Indices[i] = perm[i];
    }

    SAParams params;
    double score = ScoreOrder(Indices, corr, params);
    cout << "RCM score=" << score << endl;
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
    Correlation corr;
    corr.Calculate(m);

    std::vector<int> Indices;
    SAParams sa_params;
    SimulatedAnnealing(corr, Indices, sa_params);

    //RCMOrder(corr, Indices);

    // Generate heatmap

    const int width = m.MatrixWidth;
    uint8_t* heatmap = SIMDSafeAllocate(width * width * 3);
    ScopedF heatmap_scope([&]() {
        SIMDSafeFree(heatmap);
    });

    for (int i = 0; i < width; ++i)
    {
        int offset = i * (i + 1) / 2;

        // Amplify all the correlations a lot since most are pretty small
        double r_norm_factor = width / 2;

        for (int j = 0; j <= i; ++j)
        {
            double r = corr.RMatrix[offset + j] * r_norm_factor;

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
