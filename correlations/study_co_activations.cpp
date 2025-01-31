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

struct ClusterSortIndicesParams
{
    int max_epochs = 100;
    float cluster_thresh = 0.05f;
    float sort_threshold = 0.05f;
    int cluster_multiple = 16;
    int max_cluster_count = 128;
};

struct Cluster
{
    // Indices that represent the cluster (distance is taken relative to these)
    std::vector<int> centroids;

    // List of neurons in this cluster
    std::vector<int> neurons;
};

static int CountCorrelatedColumns(
    int i,
    const std::vector<int>& neurons_j,
    Correlation& corr,
    float cluster_thresh)
{
    int edge_count = 0;

    const int cj_count = (int)neurons_j.size();
    const int max_edge_check = 64;

    int cj = 0;
    for (; cj < cj_count && cj < max_edge_check; ++cj) {
        float r = corr.Get(i, neurons_j[cj]);
        if (r > cluster_thresh) {
            edge_count++;
        }
    }

    // Skip middle
    cj = std::max(cj_count - max_edge_check, cj);

    for (; cj < cj_count; ++cj) {
        float r = corr.Get(i, neurons_j[cj]);
        if (r > cluster_thresh) {
            edge_count++;
        }
    }

    return edge_count;
}

// Rate of the value of merging two clusters from 0..1
static float ScoreClusterMerge(
    Correlation& corr,
    const std::vector<int>& neurons_i,
    const std::vector<int>& neurons_j,
    float cluster_thresh = -2.f)
{
    int edge_count = 0;

    const int ci_count = (int)neurons_i.size();
    const int cj_count = (int)neurons_j.size();
    const int max_edge_check = 64;

    // If it is worth the time to cut out the middle:
    if (ci_count > max_edge_check * 3 || cj_count > max_edge_check * 3) {
        int ci = 0;
        for (; ci < ci_count && ci < max_edge_check; ++ci) {
            const int i = neurons_i[ci];

            edge_count += CountCorrelatedColumns(i, neurons_j, corr, cluster_thresh);
        }

        // Skip middle
        ci = std::max(ci_count - max_edge_check, ci);

        for (; ci < ci_count; ++ci) {
            const int i = neurons_i[ci];

            edge_count += CountCorrelatedColumns(i, neurons_j, corr, cluster_thresh);
        }
    } else {
        for (int ci : neurons_i) {
            for (int cj : neurons_j) {
                float r = corr.Get(ci, cj);
                if (r > cluster_thresh) {
                    edge_count++;
                }
            }
        }
    }

    // What percentage of the combined space would be filled?
    const float score = edge_count / static_cast<float>( 2 * ci_count * cj_count );

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

static void SortCluster(Correlation& corr, std::shared_ptr<Cluster>& cluster, float r_thresh) {
    const int cluster_neuron_count = static_cast<int>(cluster->neurons.size());
    cout << "Aggressively sorting cluster with " << cluster_neuron_count << " neurons..." << endl;

    int loss = NeuronSortL1Loss(cluster->neurons, corr, r_thresh);
    int move_distance = cluster_neuron_count / 2;

    while (move_distance > 0) {
        bool no_change = true;

        for (int i = move_distance; i < cluster_neuron_count; ++i) {
            // Perform the move
            int neuron_to_move = cluster->neurons[i];
            cluster->neurons.erase(cluster->neurons.begin() + i);
            cluster->neurons.insert(cluster->neurons.begin() + i - move_distance, neuron_to_move);

            // Calculate loss after the move
            int new_loss = NeuronSortL1Loss(cluster->neurons, corr, r_thresh);

            if (new_loss < loss) {
                // If the new configuration is better, update the loss and mark as change
                loss = new_loss;
                no_change = false;
            } else {
                // Move was not beneficial, reverse it
                cluster->neurons.erase(cluster->neurons.begin() + i - move_distance);
                cluster->neurons.insert(cluster->neurons.begin() + i, neuron_to_move);
            }
        }

        if (no_change) {
            move_distance /= 2; // More aggressive reduction of move_distance
        }
    }
}

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
                score = ScoreClusterMerge(corr, cluster_i->neurons, cluster_j->neurons, params.cluster_thresh);
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
                if ((int)clusters[i]->neurons.size() + (int)clusters[j]->neurons.size() > params.max_cluster_count) {
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
                score = ScoreClusterMerge(corr, cluster_i->neurons, cluster_k->neurons, params.cluster_thresh);
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
            float score = ScoreClusterMerge(corr, tiny_cluster->neurons, cluster_j->neurons, params.cluster_thresh);

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

    // Sort clusters

    for (auto& cluster : clusters) {
        SortCluster(corr, cluster, params.sort_threshold);
    }

    // Produce final cluster scores

    // Lower triangular score matrix
    std::vector<float> final_pair_scores(cluster_count * (cluster_count + 1) / 2);

    for (int i = 0; i < cluster_count; ++i) {
        auto& cluster_i = clusters[i];
        int row_offset = i * (i + 1) / 2;
        for (int j = 0; j < i; ++j) {
            auto& cluster_j = clusters[j];
            float score = ScoreClusterMerge(corr, cluster_i->neurons, cluster_j->neurons, params.cluster_thresh);
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
            cout << "No pairs found" << endl;
            break;
        }

        auto cluster_i = clusters[max_pair_i];
        auto cluster_j = clusters[max_pair_j];

        --clusters_remaining;
        cout << "Merging cluster (score=" << max_pair_score << ") " << max_pair_i << " with " << max_pair_j << " ("
            << cluster_i->neurons.size() << "+" << cluster_j->neurons.size() << " neurons): "
            << clusters_remaining << " clusters remain" << endl;

        MergeWithBestClusterOrder(cluster_i->neurons, cluster_j->neurons, corr, params.sort_threshold);

        final_eliminated_clusters[max_pair_j] = true;

        // Update scores just for the merged cluster

        for (int k = 0; k < cluster_count; ++k) {
            if (final_eliminated_clusters[k]) {
                continue;
            }
            auto& cluster_k = clusters[k];

            float score = ScoreClusterMerge(corr, cluster_i->neurons, cluster_k->neurons, params.cluster_thresh);

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

static void GenerateNeuronHistogram(CorrelationMatrix& m, const std::string& filename)
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
    cv::imwrite(filename, hist_image);
}

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

static void StudyCoactivations(
    CorrelationMatrix& m,
    Correlation& corr,
    std::vector<int>& indices,
    std::vector<uint8_t>& above_count,
    std::vector<uint8_t>& below_count,
    int neighbor_limit,
    float r_thresh)
{
    if (neighbor_limit > 255) {
        throw std::runtime_error("Neighbor limit too high");
    }

    const int width = m.MatrixWidth;

    corr.Calculate(m);

    indices.resize(width);
    for (int i = 0; i < width; ++i) {
        indices[i] = i;
    }
    cout << "Unordered score=" << ScoreOrder(width, indices.data(), corr) << endl;

    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return m.Get(i, i) < m.Get(j, j);
    });
    cout << "Sorted score=" << ScoreOrder(width, indices.data(), corr) << endl;

    indices = ClusterSortIndices(corr);

    double score = ScoreOrder(width, indices.data(), corr);
    cout << "Final score=" << score << endl;

    above_count.resize(width);
    below_count.resize(width);

    for (int i = 0; i < width; ++i)
    {
        const int neuron_i = indices[i];

        int below = 0;

        for (int j = 1; j < neighbor_limit; ++j)
        {
            const int neighbor_j = i - j;
            if (neighbor_j < 0) {
                break;
            }
            const int neighbor_neuron_j = indices[neighbor_j];

            const float r = corr.Get(neuron_i, neighbor_neuron_j);
            if (r > r_thresh) {
                below = j;
            }
        }

        below_count[i] = static_cast<uint8_t>( below );

        int above = 0;

        for (int j = 1; j < neighbor_limit; ++j)
        {
            const int neighbor_j = i + j;
            if (neighbor_j >= width) {
                break;
            }
            const int neighbor_neuron_j = indices[neighbor_j];

            const float r = corr.Get(neuron_i, neighbor_neuron_j);
            if (r > r_thresh) {
                above = j;
            }
        }

        above_count[i] = static_cast<uint8_t>( above );

        cout << "Neuron " << i << " : below=" << below << " above=" << above << endl;
    }
}

static void GenerateHeatmap(Correlation& corr, std::vector<int>& indices, const std::string& filename)
{
    const int width = corr.Width;

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
    cv::imwrite(filename, heatmap_image);
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

    Correlation corr;
    std::vector<int> indices;
    std::vector<uint8_t> above_count;
    std::vector<uint8_t> below_count;

    const int neighbor_limit = 64;
    const float r_thresh = 0.05;

    StudyCoactivations(m, corr, indices, above_count, below_count, neighbor_limit, r_thresh);

    std::string hist_filename = "histogram_block_";
    hist_filename += std::to_string(m.BlockNumber);
    hist_filename += ".png";

    GenerateNeuronHistogram(m, hist_filename);

    std::string heatmap_filename = "heatmap_block_";
    heatmap_filename += std::to_string(m.BlockNumber);
    heatmap_filename += ".png";

    GenerateHeatmap(corr, indices, heatmap_filename);

    return 0;
}
