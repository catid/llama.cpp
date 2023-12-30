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

static void SimulatedAnnealing(std::vector<int>& indices, Correlation& corr, const SAParams& params)
{
    const int width = (int)indices.size();

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

        //double score = ScoreOrder(width, indices.data(), corr);
        //cout << "Epoch " << epoch << " score=" << score << endl;
    }
}

/*
    Fiedler Partitioning method: https://shainarace.github.io/LinearAlgebra/chap1-5.html
    This algorithm is too slow to look for graph cuts, so we only run it on sub-graphs (dim<4000).

    Rather than looking for graph cuts, we simply calculate the Fiedler vector (second Eigenvector),
    and we observe that sorting the indices will also sort the neurons by connectivity.

    Provided with a subset of the indices to sort, and the correlation matrix for lookups,
    we sort the indices in-place.
*/
static bool FiedlerSort(std::vector<int>& indices, Correlation& corr, float threshold)
{
    // Only need to do lower triangle
    const int width = (int)indices.size();
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

struct ClusterSortIndicesParams
{
    int max_epochs = 100;
    float cluster_thresh = 0.05f;
    float sort_threshold = 0.05f;
    int cluster_multiple = 16;
    int max_cluster_count = 64;
};

struct Cluster
{
    // Indices that represent the cluster (distance is taken relative to these)
    std::vector<int> centroids;

    // List of neurons in this cluster
    std::vector<int> neurons;
};

// Rate of the value of merging two clusters from 0..1
static float ScoreClusterMerge(
    Correlation& corr,
    const std::shared_ptr<Cluster>& cluster_i,
    const std::shared_ptr<Cluster>& cluster_j,
    float cluster_thresh = -2.f)
{
    int edge_count = 0;

    for (int ci : cluster_i->neurons) {
        for (int cj : cluster_j->neurons) {
            float r = corr.Get(ci, cj);
            if (r > cluster_thresh) {
                edge_count++;
            }
        }
    }

    const int max_score = 2 * (int)cluster_i->neurons.size() * (int)cluster_j->neurons.size();

    // What percentage of the combined space is filled?
    const float score = edge_count / static_cast<float>( max_score );

    return score;
}

static bool containsAllValues(const std::vector<int>& vec) {
    int N = vec.size();
    std::vector<bool> seen(N, false);

    for (int num : vec) {
        if (num < 0 || num >= N) {
            // Number out of range
            return false;
        }
        if (seen[num]) {
            // Duplicate number found
            return false;
        }
        seen[num] = true;
    }

    // If any number was not seen, return false
    for (bool flag : seen) {
        if (!flag) return false;
    }

    return true;
}

template <typename T>
static size_t argmax(const std::vector<T>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

/*
    Possible ways to combine the two:

    score0: 123 456
    score1: 123 654
    score2: 321 456
    score3: 321 654

    Try all 4 and see which one scores the best, and merge cluster1 into cluster0 using this order.
*/
static void MergeWithBestClusterOrder(std::vector<int>& cluster0, std::vector<int>& cluster1, Correlation& corr, float r_thresh)
{
    const int width0 = (int)cluster0.size();
    const int width1 = (int)cluster1.size();

    std::vector<int> scores(4);

    for (int i = 0; i < width0; ++i) {
        const int neuron0 = cluster0[i];

        for (int j = 0; j < width1; ++j) {
            const int neuron1 = cluster1[j];

            float r = corr.Get(neuron0, neuron1);
            if (r < r_thresh) {
                continue;
            }

            scores[3] += (width0-1 - i) + j + 1;
            scores[2] += (width0-1 - i) + (width0-1 - j);
            scores[1] += i + (width1-1 - j) + 1;
            scores[0] += i + j + 1;
        }
    }

    int best_score = argmax(scores);
    switch (best_score)
    {
    default:
    case 0: // score0: 123 456
        break;
    case 1: // score1: 123 654
        std::reverse(cluster1.begin(), cluster1.end());
        break;
    case 2: // score2: 321 456
        std::reverse(cluster0.begin(), cluster0.end());
        break;
    case 3: // score3: 321 654
        std::reverse(cluster0.begin(), cluster0.end());
        std::reverse(cluster1.begin(), cluster1.end());
        break;
    }

    cluster0.insert(cluster0.end(), cluster1.begin(), cluster1.end());
}

static int NeuronSortL1Loss(std::vector<int>& neurons, Correlation& corr, float r_thresh)
{
    int score = 0;

    const int width = (int)neurons.size();
    for (int i = 0; i < width; ++i) {
        const int row_i = neurons[i];
        for (int j = 0; j < i; ++j) {
            const int col_j = neurons[j];

            float r = corr.Get(row_i, col_j);
            if (r < r_thresh) {
                continue;
            }

            score += i - j - 1;
        }
    }

    return score;
}

#if 0

static void SortCluster(Correlation& corr, std::shared_ptr<Cluster>& cluster, float r_thresh)
{
    const int cluster_neuron_count = static_cast<int>( cluster->neurons.size() );

    // Find the best neighbor swaps that improve score

    int loss = NeuronSortL1Loss(cluster->neurons, corr, r_thresh);

    int swap_distance = cluster_neuron_count / 2 + 1;
    if (swap_distance >= cluster_neuron_count) {
        swap_distance = cluster_neuron_count - 1;
    }

    for (; swap_distance >= 1; --swap_distance)
    {
        for (int trials = 10; trials >= 0; --trials)
        {
            bool no_change = true;

            for (int i = swap_distance; i < cluster_neuron_count; ++i) {
                std::swap(cluster->neurons[i - swap_distance], cluster->neurons[i]);

                int swap_loss = NeuronSortL1Loss(cluster->neurons, corr, r_thresh);

                if (swap_loss > loss) {
                    std::swap(cluster->neurons[i - swap_distance], cluster->neurons[i]);
                } else {
                    loss = swap_loss;
                    no_change = false;
                }
            }

            if (no_change) {
                break;
            }
        }
    }
}

#else

static int HypotheticalSwapLoss(int idx1, int idx2, std::vector<int>& neurons, Correlation& corr, float r_thresh) {
    int scoreChange = 0;

    // Calculate the contribution to the score from the current positions
    for (int i = 0; i < neurons.size(); ++i) {
        if (i == idx1 || i == idx2) continue;
        const int currentIdx = (i < idx1) ? i : idx1;
        const int swapIdx = (i < idx2) ? i : idx2;

        float rCurrent = corr.Get(neurons[idx1], neurons[currentIdx]);
        float rSwap = corr.Get(neurons[idx2], neurons[swapIdx]);

        if (rCurrent >= r_thresh) {
            scoreChange -= std::abs(idx1 - i) - 1;
        }
        if (rSwap >= r_thresh) {
            scoreChange += std::abs(idx2 - i) - 1;
        }
    }

    // Factor in the direct swap between idx1 and idx2
    float rDirectSwap = corr.Get(neurons[idx1], neurons[idx2]);
    if (rDirectSwap >= r_thresh) {
        scoreChange += std::abs(idx1 - idx2) - 1;  // Adding the score as they are now in correct order
    }

    return NeuronSortL1Loss(neurons, corr, r_thresh) + scoreChange;
}

static void SortCluster(Correlation& corr, std::shared_ptr<Cluster>& cluster, float r_thresh) {
    const int cluster_neuron_count = static_cast<int>(cluster->neurons.size());

    int loss = NeuronSortL1Loss(cluster->neurons, corr, r_thresh);
    int swap_distance = cluster_neuron_count / 2;

    while (swap_distance > 0) {
        bool no_change = true;

        for (int i = swap_distance; i < cluster_neuron_count; ++i) {
            // Hypothetical swap loss calculation (modifying NeuronSortL1Error might be necessary)
            int hypothetical_loss = HypotheticalSwapLoss(i, i - swap_distance, cluster->neurons, corr, r_thresh);

            if (hypothetical_loss < loss) {
                std::swap(cluster->neurons[i - swap_distance], cluster->neurons[i]);
                loss = hypothetical_loss;
                no_change = false;
            }
        }

        if (no_change) {
            swap_distance /= 2; // More aggressive reduction of swap_distance
        }
    }
}

#endif

static std::vector<int> ClusterSortIndices(Correlation& corr, const ClusterSortIndicesParams& params = ClusterSortIndicesParams())
{
    const int neuron_count = corr.Width;

    // Count the number of edges for each node

    std::vector<int> edge_count(neuron_count);

    for (int i = 0; i < neuron_count; ++i) {
        int count = 0;
        for (int j = 0; j < neuron_count; ++j) {
            if (i == j) {
                continue;
            }

            float r = corr.Get(i, j);
            if (r < params.cluster_thresh) {
                continue;
            }

            ++count;
        }

        edge_count[i] = count;
    }

    // Sort neurons by edge count

    std::vector<int> edge_sorted_neurons(neuron_count);
    for (int i = 0; i < neuron_count; ++i) {
        edge_sorted_neurons[i] = i;
    }
    std::sort(edge_sorted_neurons.begin(), edge_sorted_neurons.end(), [&](int i, int j) {
        return edge_count[i] > edge_count[j];
    });

    // Pick the most disjoint but connected neurons as initial cluster centers

    std::vector<std::shared_ptr<Cluster>> clusters;
    std::vector<bool> is_centroid(neuron_count);

    for (int sort_i = 0; sort_i < neuron_count; ++sort_i) {
        int i = edge_sorted_neurons[sort_i];

        bool found = false;
        for (auto& cluster : clusters) {
            for (int j : cluster->centroids) {
                float r = corr.Get(i, j);
                if (r > params.cluster_thresh) {
                    found = true;
                    break;
                }
                if (found) {
                    break;
                }
            }
        }

        if (!found) {
            auto cluster = std::make_shared<Cluster>();
            cluster->centroids.push_back(i);
            cluster->neurons.push_back(i);
            clusters.push_back(cluster);

            is_centroid[i] = true;
        }
    }

    int cluster_count = static_cast<int>( clusters.size() );
    cout << "Found " << cluster_count << " clusters" << endl;

    // Assign all the remaining neurons to nearest clusters:

    for (int sort_i = 0; sort_i < neuron_count; ++sort_i)
    {
        int i = edge_sorted_neurons[sort_i];

        if (is_centroid[i]) {
            continue;
        }

        double max_score = -1.0;
        int max_score_cluster = -1;

        for (int j = 0; j < cluster_count; ++j) {
            auto& cluster_j = clusters[j];

            // Skip clusters that are already full
            if ((int)cluster_j->neurons.size() >= params.max_cluster_count) {
                continue;
            }

            double score = 0.0;
            for (int k : cluster_j->centroids) {
                score += corr.Get(i, k);
            }

            if (max_score < score) {
                max_score = score;
                max_score_cluster = j;
            }
        }

        // If there is no good match:
        if (max_score_cluster < 0.0) {
            // Just make a new cluster for this neuron
            auto cluster = std::make_shared<Cluster>();
            cluster->centroids.push_back(i);
            cluster->neurons.push_back(i);
            clusters.push_back(cluster);

            is_centroid[i] = true;
            ++cluster_count;
        } else {
            clusters[max_score_cluster]->neurons.push_back(i);
        }
    }

    cout << "Assigned all neurons to " << cluster_count << " clusters" << endl;

    // Split off all clusters with a tiny number of neurons

    std::vector<std::shared_ptr<Cluster>> tiny_clusters;

    for (int i = 0; i < cluster_count; ++i) {
        auto& cluster_i = clusters[i];
        if (cluster_i->neurons.size() >= 3) {
            continue;
        }

        tiny_clusters.push_back(cluster_i);

        clusters.erase(clusters.begin() + i);
        --cluster_count;
        --i;
    }

    cout << "Split off tiny clusters leaving " << cluster_count << " popular clusters" << endl;

    // Produce initial cluster scores:

    // Lower triangular score matrix
    std::vector<float> cluster_pair_scores(cluster_count * (cluster_count + 1) / 2);

    for (int i = 0; i < cluster_count; ++i) {
        auto& cluster_i = clusters[i];
        int row_offset = i * (i + 1) / 2;
        for (int j = 0; j < i; ++j) {
            auto& cluster_j = clusters[j];

            float score = -1;
            if ((int)cluster_i->neurons.size() + (int)cluster_j->neurons.size() <= params.max_cluster_count) {
                score = ScoreClusterMerge(corr, cluster_i, cluster_j, params.cluster_thresh);
            }

            cluster_pair_scores[row_offset + j] = score;
        }
    }

    std::vector<bool> eliminated_clusters(cluster_count);
    int clusters_remaining = cluster_count;

    for (int pair_index = 0;; ++pair_index)
    {
        // Find largest pair score:

        float max_pair_score = -2.f;
        int max_pair_i = -1, max_pair_j = -1;

        for (int i = 0; i < cluster_count; ++i) {
            int row_offset = i * (i + 1) / 2;
            if (eliminated_clusters[i]) {
                continue;
            }
            for (int j = 0; j < i; ++j) {
                if (eliminated_clusters[j]) {
                    continue;
                }
                if ((int)clusters[i]->neurons.size() + (int)clusters[i]->neurons.size() > params.max_cluster_count) {
                    continue;
                }
                const float score = cluster_pair_scores[row_offset + j];
                if (max_pair_score < score) {
                    max_pair_score = score;
                    max_pair_i = i;
                    max_pair_j = j;
                }
            }
        }

        // If no scores are good, meaning no clusters should be merged:
        if (max_pair_score <= 0.0f) {
            break;
        }

        auto cluster_i = clusters[max_pair_i];
        auto cluster_j = clusters[max_pair_j];

        cluster_i->centroids.insert(cluster_i->centroids.end(), cluster_j->centroids.begin(), cluster_j->centroids.end());
        cluster_i->neurons.insert(cluster_i->neurons.end(), cluster_j->neurons.begin(), cluster_j->neurons.end());

        --clusters_remaining;
        cout << "Merged cluster (score=" << max_pair_score << ") " << max_pair_i << " with " << max_pair_j << " (" << cluster_i->neurons.size() << " neurons): " << clusters_remaining << " clusters remain" << endl;

        eliminated_clusters[max_pair_j] = true;

        // Update scores just for the merged cluster

        for (int k = 0; k < cluster_count; ++k) {
            if (eliminated_clusters[k]) {
                continue;
            }
            auto& cluster_k = clusters[k];

            float score = -2;

            const int combined_size = (int)cluster_i->neurons.size() + (int)cluster_k->neurons.size();

            if (combined_size <= 64) {
                score = ScoreClusterMerge(corr, cluster_i, cluster_k, params.cluster_thresh);
            }

            if (k < max_pair_i) {
                // Store at (i, k)
                cluster_pair_scores[max_pair_i * (max_pair_i + 1) / 2 + k] = score;
            } else {
                // Store at (k, i)
                cluster_pair_scores[k * (k + 1) / 2 + max_pair_i] = score;
            }
        }
    }

    // Remove dead clusters

    for (int i = cluster_count - 1; i >= 0; --i) {
        if (eliminated_clusters[i]) {
            clusters.erase(clusters.begin() + i);
        }
    }
    cluster_count = static_cast<int>( clusters.size() );

    cout << "Merged popular clusters into " << cluster_count << " clusters" << endl;

    // Merge tiny clusters into popular clusters

    for (auto& tiny_cluster : tiny_clusters) {
        float max_score = -2.f;
        int max_score_cluster = -1;

        for (int j = 0; j < cluster_count; ++j) {
            auto& cluster_j = clusters[j];

            // Skip clusters that are already full
            if ((int)cluster_j->neurons.size() >= params.max_cluster_count) {
                continue;
            }

            // Check how correlated two clusters are
            float score = ScoreClusterMerge(corr, tiny_cluster, cluster_j, params.cluster_thresh);

            if (max_score < score) {
                max_score = score;
                max_score_cluster = j;
            }
        }

        if (max_score < 0.f || max_score_cluster < 0) {
            // Just append it
            clusters.push_back(tiny_cluster);
            continue;
        }

        auto& cluster_dest = clusters[max_score_cluster];

        cluster_dest->neurons.insert(cluster_dest->neurons.end(), tiny_cluster->neurons.begin(), tiny_cluster->neurons.end());
    }
    cluster_count = static_cast<int>( clusters.size() );

    cout << "Merged " << tiny_clusters.size() << " tiny clusters into " << cluster_count << " final clusters" << endl;

    // Produce final cluster scores:

    // Lower triangular score matrix
    std::vector<float> final_pair_scores(cluster_count * (cluster_count + 1) / 2);

    for (int i = 0; i < cluster_count; ++i) {
        auto& cluster_i = clusters[i];
        int row_offset = i * (i + 1) / 2;
        for (int j = 0; j < i; ++j) {
            auto& cluster_j = clusters[j];
            float score = ScoreClusterMerge(corr, cluster_i, cluster_j, params.cluster_thresh);
            final_pair_scores[row_offset + j] = score;
        }
    }

    std::vector<bool> final_eliminated_clusters(cluster_count);
    clusters_remaining = cluster_count;

    for (int pair_index = 0;; ++pair_index)
    {
        // Find largest pair score:

        float max_pair_score = -2.f;
        int max_pair_i = -1, max_pair_j = -1;

        for (int i = 0; i < cluster_count; ++i) {
            int row_offset = i * (i + 1) / 2;
            if (final_eliminated_clusters[i]) {
                continue;
            }
            for (int j = 0; j < i; ++j) {
                if (final_eliminated_clusters[j]) {
                    continue;
                }
                const float score = final_pair_scores[row_offset + j];
                if (max_pair_score < score) {
                    max_pair_score = score;
                    max_pair_i = i;
                    max_pair_j = j;
                }
            }
        }

        // If no scores are good, meaning no clusters should be merged:
        if (max_pair_score <= 0.0f) {
            break;
        }

        auto cluster_i = clusters[max_pair_i];
        auto cluster_j = clusters[max_pair_j];

        --clusters_remaining;
        cout << "Merging cluster (score=" << max_pair_score << ") " << max_pair_i << " with " << max_pair_j << " ("
            << cluster_i->neurons.size() << "+" << cluster_j->neurons.size() << " neurons): "
            << clusters_remaining << " clusters remain" << endl;

        // Sort small cluster-pairs
        if ((int)cluster_i->neurons.size() <= params.max_cluster_count*2 + 8) {
            cout << "Thoroughly sorting small merged cluster..." << endl;

            cluster_i->neurons.insert(cluster_i->neurons.end(), cluster_j->neurons.begin(), cluster_j->neurons.end());

            SortCluster(corr, cluster_i, params.sort_threshold);
        } else {
            MergeWithBestClusterOrder(cluster_i->neurons, cluster_j->neurons, corr, params.sort_threshold);
        }

        final_eliminated_clusters[max_pair_j] = true;

        // Update scores just for the merged cluster

        for (int k = 0; k < cluster_count; ++k) {
            if (final_eliminated_clusters[k]) {
                continue;
            }
            auto& cluster_k = clusters[k];

            float score = -2;

            const int combined_size = (int)cluster_i->neurons.size() + (int)cluster_k->neurons.size();

            if (combined_size <= 64) {
                score = ScoreClusterMerge(corr, cluster_i, cluster_k, params.cluster_thresh);
            }

            if (k < max_pair_i) {
                // Store at (i, k)
                final_pair_scores[max_pair_i * (max_pair_i + 1) / 2 + k] = score;
            } else {
                // Store at (k, i)
                final_pair_scores[k * (k + 1) / 2 + max_pair_i] = score;
            }
        }
    }

    // Remove dead clusters

    for (int i = cluster_count - 1; i >= 0; --i) {
        if (final_eliminated_clusters[i]) {
            clusters.erase(clusters.begin() + i);
        }
    }
    cluster_count = static_cast<int>( clusters.size() );

    cout << "Merged final clusters into " << cluster_count << " clusters" << endl;

    // Produce final list

    std::vector<int> sorted_indices;
    for (int i = 0; i < cluster_count; ++i) {
        auto& cluster = clusters[i];
        sorted_indices.insert(sorted_indices.end(), cluster->neurons.begin(), cluster->neurons.end());
    }

    if (!containsAllValues(sorted_indices)) {
        cout << "FIXME: containsAllValues = false" << endl;
    }

    cout << "Sorting " << sorted_indices.size() << " neurons complete" << endl;

    return sorted_indices;
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

    indices = ClusterSortIndices(corr);

#if 0
    SAParams sa_params;
    sa_params.max_move = m.MatrixWidth / 8;
    sa_params.max_epochs = 100;
    SimulatedAnnealing(indices, corr, sa_params);
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
