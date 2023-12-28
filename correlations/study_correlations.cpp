#include "compress.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Tried MKL version and it fails(!)
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
using namespace Eigen;

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

    // Lower triangular correlation matrix
    float* RMatrix = nullptr;
    float LargestR = 0.0;

    // List of auto-encoder classified neurons.
    // This will also contain neurons that never were observed firing.
    std::vector<int> AutoEncoderNeurons;

    int AutoEncoderNeuronCount = 0;
    int KnowledgeNeuronCount = 0;

    // Maximum seen histogram count for knowledge-classified neurons
    uint32_t MaxKnowledgeHistValue = 0;

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

                    This means we just need a lower-triangular correlation matrix (half the memory).
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
        } // next column
    } // next row
}

/*
    Fiedler Partitioning method: https://shainarace.github.io/LinearAlgebra/chap1-5.html
    This algorithm is too slow to look for graph cuts, so we only run it on sub-graphs (dim<4000).

    Rather than looking for graph cuts, we simply calculate the Fiedler vector (second Eigenvector),
    and we observe that sorting the indices will also sort the neurons by connectivity.

    Provided with a subset of the indices to sort, and the correlation matrix for lookups,
    we sort the indices in-place.
*/
static bool FiedlerSort(int width, int* indices, Correlation& corr, float threshold)
{
    // Only need to do lower triangle
    MatrixXf A(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < i; ++j) {
            const float r = corr.Get(indices[i], indices[j]);
            A(i, j) = r > threshold ? 1.f : 0.f;
        }
    }

    VectorXf degrees = A.rowwise().sum();
    MatrixXf D = degrees.asDiagonal();
    MatrixXf L = D - A;
    MatrixXd L_d = L.cast<double>();

    SelfAdjointEigenSolver<MatrixXd> eigensolver(L_d);
    if (eigensolver.info() != Success) {
        // Handle the error appropriately
        cerr << "Eigenvalue decomposition failed." << endl;
        return false;
    }

    VectorXd fiedler_vector = eigensolver.eigenvectors().col(1);

    using pair_i2 = std::pair<int, int>;
    vector<pair_i2> order(width);
    for (int i = 0; i < width; ++i) {
        order[i] = std::make_pair(i, indices[i]);
    }

    sort(order.begin(), order.end(),
        [&](const pair_i2& i, const pair_i2& j) {
            return fiedler_vector(i.first) < fiedler_vector(j.first);
        });

    for (int i = 0; i < width; ++i) {
        indices[i] = order[i].second;
    }

    return true;
}

struct KMeansParams
{
    int max_epochs = 100;
    float r_thresh = 0.05f;
    int diag_dist_score = 32;
};

static std::vector<int> KMeansReorder(int width, Correlation& corr, const KMeansParams& params = KMeansParams())
{
    std::vector<int> indices(width), scores(width), centroids;

    for (int epoch = 0; epoch < params.max_epochs; ++epoch)
    {
        for (int i = 0; i < width; ++i) {
            // We just count the number of correlated neurons
            int score = 0;
            for (int k = 0; k < width; ++k) {
                float r = corr.Get(i, k);
                if (r < params.r_thresh) {
                    continue;
                }
                ++score;
            }

            indices[i] = i;
            scores[i] = score;
        }

        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return scores[i] < scores[j];
        });

        const int largest = indices[0];
    }

    return indices;
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

    SAParams sa_params;
    sa_params.max_move = m.MatrixWidth / 8;
    indices = SimulatedAnnealing(corr, sa_params);

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
            if (r < 0.05) {
                r = 0.0;
            }

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
