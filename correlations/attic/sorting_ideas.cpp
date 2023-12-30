
struct SAParams
{
    int max_move = 5000;
    int max_epochs = 1000;
    float r_thresh = 0.05f;
};

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
