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

    /*
        Correlation matrix is not symmetric.  For example P(J|I) != P(I|J)
        meaning if neuron J fires, it may mean I fires more often than average,
        but the reverse may not be true: If neuron I fires it may not cause
        neuron J to fire any more often than normal.

        If I is a row and J is a column, we store the matrix row-first:
        The upper triangular matrix where J > I is storing P(J|I) - P(J).
        The lower triangular matrix where I < J is storing P(I|J) - P(I).

        So  "I implies J" or I -> J is in the upper right.
        And "J implies I" or J -> I is in the lower left.
    */
    float* RMatrix = nullptr;
    float LargestR = 0.0;

    // List of auto-encoder classified neurons.
    // This will also contain neurons that never were observed firing.
    std::vector<int> AutoEncoderNeurons;

    int AutoEncoderNeuronCount = 0;
    int KnowledgeNeuronCount = 0;

    // Maximum seen histogram count for knowledge-classified neurons
    uint32_t MaxKnowledgeHistValue = 0;

    // See the comment above for RMatrix to understand what this is.
    float Get(int i/*row*/, int j/*column*/) {
        if (j > i) {
            std::swap(i, j);
        }
        const int row_offset = i * (i + 1) / 2;
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

#if 0
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
#endif

    const double inv_total = 1.0 / (double)m.TotalTrials;

    for (int i = 0; i < width; ++i)
    {
        const int row_offset = i * (i + 1) / 2;

        // Calculate P(I) = Hist(I,I) / TotalCount
        const uint32_t hist_i = m.Get(i, i);
        const double p_i = hist_i * inv_total;

        // Note that for the diagonal this produces a correlation of 1.0 as expected
        for (int j = 0; j <= i; ++j)
        {
            const uint32_t hist_ij = m.Get(i, j);
            const uint32_t hist_j = m.Get(j, j);

            if (hist_j == 0 || hist_i == 0) {
                RMatrix[row_offset + j] = -1.f;
                continue;
            }

            /*
                What is conditional P(I | J) - P(I) ?

                * It ranges from -1 to 1 just like a correlation.
                * It represents "how much more" event I happens than average when J is happening.
                * This measures how true is the logical statement "I implies J", so if it is -1
                  then "I implies Not J" and if it is 1, then "I implies J" 100% of the time.

                How does this compare to P(J | I) - P(J) ?

                * It's the logical implication I -> J ("I implies J") instead of ("J implies I").
                * Otherwise it's the same idea.  We store it in the correlation matrix on the
                  opposite side of the diagonal so both can be looked up later.
            */

            /*
                If I is a row and J is a column, we store the matrix row-first:

                    The lower triangular matrix where I<J is storing P(I|J) - P(I).
                    The upper triangular matrix where J>I is storing P(J|I) - P(J).

                    So "J implies I" or J->I is in the lower left.
                    So "I implies J" or I->J is in the upper right.

                Observations:

                    Sort the neurons by firing frequency.  Then:

                    The upper right triangle is a lot less cloudy than the lower left (particularly at the bottom).
                    So you get better signal from Row activation -> Column activation than the other way.

                    In fact we don't even care about the bottom triangle data at all since it's noisy.

                Conclusion:

                    Rather than storing both I->J and J->I correlations:

                    Choose P(J|I) - P(J), for P(J) < P(I).
                    Choose P(I|J) - P(I), for P(I) > P(J).
            */

            // Hist(J) < Hist(I) implies that P(J) < P(I)
            if (hist_j < hist_i) {
                // Calculate P(J) = Hist(J,J) / TotalCount
                const double p_j = hist_j * inv_total;

                // Calculate P(J|I) = Hist(I,J) / Hist(J,J) = Hist(J,I) / Hist(J,J)
                const double cond_p_j_given_i = hist_ij / (double)hist_i;

                // Choose C(I,J) = P(J|I) - P(J), for P(J) > P(I).
                RMatrix[row_offset + j] = static_cast<float>( cond_p_j_given_i - p_j );
            } else {
                // Calculate conditional P(I | J) = P(J and I) / P(J) from histogram counts:
                //  P(J) = Hist(J,J) / TotalCount
                //  P(J and I) = Hist(I,J) / TotalCount = Hist(J,I) / TotalCount
                //  So the "/ TotalCount" factor cancels out: P(I|J) = Hist(I,J) / Hist(J,J)
                const double cond_p_i_given_j = hist_ij / (double)hist_j;

                // Choose C(I,J) = P(I|J) - P(I), for P(I) < P(J).
                RMatrix[row_offset + j] = static_cast<float>( cond_p_i_given_j - p_i );
            }
        }
    }

#if 0

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

#endif
}

struct SAParams
{
    int max_move = 5000;
    int max_epochs = 1000;
    float r_thresh = 0.05f;
};

static double ScoreOrder(int width, const int* indices, Correlation& corr, int diag_dist_score = 32)
{
    double score = 0.0;

    // For just everything under the diagonal:
    for (int i = 0; i < width; ++i)
    {
        const int row_i = indices[i];

        for (int j = 0; j < i; ++j)
        {
            // Get correlation matrix entry for transformed indices,
            // which does not change the correlation matrix values but just re-orders them.
            // These are guaranteed to be unique.
            const int col_j = indices[j];

            if (row_i == col_j || row_i >= width || col_j >= width) {
                //cout << "ERROR: " << row_i << ", " << col_j << endl;
            }

            // distance from diagonal
            int d = i - j;
            if (d <= diag_dist_score) {
                float r = corr.Get(row_i, col_j);
                if (r > 0.f) {
                    score += r;
                }
            }
        }
    }

    return score;
}

static int GetDirection(int width, int i, const int* indices, Correlation& corr, const SAParams& params)
{
    const int row_i = indices[i];

    double p_sum = 0.0;
    for (int j = 0; j < i; ++j) {
        const int col_j = indices[j];

        float r = corr.Get(row_i, col_j);
        if (r < params.r_thresh) {
            continue;
        }

        double dr = static_cast<double>( r );
        p_sum += dr;
    }

    double n_sum = 0.0;
    for (int j = i + 1; j < width; ++j) {
        const int col_j = indices[j];

        float r = corr.Get(row_i, col_j);
        if (r < params.r_thresh) {
            continue;
        }

        double dr = static_cast<double>( r );
        n_sum += dr;
    }

    return p_sum > n_sum ? -1 : 1;
}

static void MoveIndex(int knowledge_count, int* indices, int i, int move)
{
    if (move > 0) {
        int end = i + move;
        if (end >= knowledge_count) {
            end = knowledge_count - 1;
        }

        const int t = indices[i];
        for (int j = i; j < end; ++j) {
            indices[j] = indices[j + 1];
        }
        indices[end] = t;
    } else {
        int end = i + move;
        if (end < 0) {
            end = 0;
        }

        const int t = indices[i];
        for (int j = i; j > end; --j) {
            indices[j] = indices[j - 1];
        }
        indices[end] = t;
    }
}

#include <random>
#include <algorithm>

static std::vector<int> SimulatedAnnealing(Correlation& corr, const SAParams& params)
{
    const int width = corr.Width;
    std::vector<int> indices(width);
    for (int i = 0; i < width; ++i) {
        indices[i] = i;
    }

    // Shuffle indices to a random initial position
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.begin() + width, g);

    uint64_t seed = g();

    // For each epoch:
    for (int epoch = 0; epoch < params.max_epochs; ++epoch)
    {
        const double temperature = 1.0 - (epoch + 1) / (double)params.max_epochs;

        // For each index:
        for (int i = 0; i < width; ++i)
        {
            int direction = GetDirection(width, i, indices.data(), corr, params);

            // Uniformly pick a random move magnitude
            int mag = params.max_move;
            mag = (uint32_t)SplitMix64(seed) % mag;

            // Modulate movement by temperature, reducing it slowly towards the end
            mag *= temperature;
            if (mag < 1) {
                mag = 1;
            }

            // Make the move
            int move = (direction > 0) ? mag : -mag;
            MoveIndex(width, indices.data(), i, move);
        }

        double score = ScoreOrder(width, indices.data(), corr);
        cout << "Epoch " << epoch << " score=" << score << endl;
    }

    return indices;
}

#if 1

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>

struct ElementInfo
{
    int Index = -1;
    float Value = 0.0;

    bool operator>(const ElementInfo& rhs) const {
        return Value > rhs.Value;
    }
};

struct RCMParams
{
    int max_k = 256; // Max count
    float min_corr = 0.07f; // Cut-off: 6% more firing than expected
    int start_index = 0;
};

static std::vector<int> RCMOrder(CorrelationMatrix& m, Correlation& corr, const RCMParams& params = RCMParams())
{
    const int width = corr.Width;

    // RCM:

    using namespace boost;
    typedef adjacency_list<vecS, vecS, undirectedS, 
        property<vertex_color_t, default_color_type,
        property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    Graph G;

    for (int i = 0; i < width; ++i) {
        // Find k largest elements
        std::priority_queue<ElementInfo, std::vector<ElementInfo>, std::greater<ElementInfo>> minHeap;

        for (int j = 0; j < i; ++j) {
            // The graph is undirected in this algorithm so use the sum of I->J and J->I correlations as an estimate.
            ElementInfo info;
            info.Value = corr.Get(i, j);
            info.Index = j;

            if (minHeap.size() < params.max_k) {
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
            if (top.Value > params.min_corr) {
                // We only consider nodes neighbors if they have positive correlation
                neighbors.push_back(top.Index);
            }
            minHeap.pop();
        }

        for (int j : neighbors) {
            //cout << "i=" << i << ", j=" << j << " -> r=" << corr.Get(i, j) << endl;
            // Edges are bidirectional, so we only need to add it once
            add_edge(i, j, G);
        }
    }

    cout << "Number of RCM graph vertices: " << num_vertices(G) << " (total neurons = " << width << ")" << endl;

    std::vector<graph_traits<Graph>::vertex_descriptor> inv_perm(num_vertices(G));

    Vertex s = vertex(params.start_index, G);

    cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G), 
                                  get(vertex_degree, G));

    // Inverse permutation to get the RCM ordering
    std::vector<graph_traits<Graph>::vertex_descriptor> perm(num_vertices(G));

    std::vector<int> indices(width);
    for (int i = 0; i < width; ++i) {
        indices[i] = i;
    }

    const int inv_perm_size = static_cast<int>( inv_perm.size() );
    for (int i = 0; i < inv_perm_size; ++i) {
        const int j = inv_perm[i];
        indices[j] = i;
        //cout << "(" << i << ", " << j << ")" << endl;
    }

    cout << "Reverse Cuthill-McKee score=" << ScoreOrder(width, indices.data(), corr) << endl;
    return indices;
}

#endif

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

    std::vector<int> indices(width);

    for (int i = 0; i < width; ++i) {
        indices[i] = i;
    }
    cout << "Unordered score=" << ScoreOrder(width, indices.data(), corr) << endl;

    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return m.Get(i, i) < m.Get(j, j);
    });
    cout << "Sorted score=" << ScoreOrder(width, indices.data(), corr) << endl;

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(indices.begin(), indices.begin() + width, g);
    cout << "Shuffle#1 score=" << ScoreOrder(width, indices.data(), corr) << endl;

    std::shuffle(indices.begin(), indices.begin() + width, g);
    cout << "Shuffle#2 score=" << ScoreOrder(width, indices.data(), corr) << endl;

#if 0
    RCMParams rcm_params;

    for (int trials = 0; trials < 100; ++trials) {
        rcm_params.start_index = g() % width;
        indices = RCMOrder(m, corr, rcm_params);
    }
#else
    SAParams sa_params;
    sa_params.max_move = m.MatrixWidth / 8;
    indices = SimulatedAnnealing(corr, sa_params);
#endif

    double score = ScoreOrder(width, indices.data(), corr);
    cout << "Final score=" << score << endl;

    // Generate heatmap

    uint8_t* heatmap = SIMDSafeAllocate(width * width * 3);
    ScopedF heatmap_scope([&]() {
        SIMDSafeFree(heatmap);
    });

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            const int row_i = indices[i];
            const int col_j = indices[j];

            // This is a value from -1..1
            //double r = corr.Get(i, j);
            float r = corr.Get(row_i, col_j);

            // Scale everything up to make things more visible
            r *= 10.f;

            // Heat should be a value from -255..255
            int heat = static_cast<int>( r * 255.f );

            int offset_ij = i * width + j;
            if (heat >= 0) {
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = (uint8_t)heat;
            }
            else if (heat < 0) {
                heat = -heat;
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = 0;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = 0;
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
