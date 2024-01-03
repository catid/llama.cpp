#include "co_activations.hpp"

#include "mmapfile.hpp"
#include "compress.hpp"

CoactivationPredictor::~CoactivationPredictor()
{
    SIMDSafeFree(Buffer);
}

bool CoactivationPredictor::ReadFile(const std::string& file_path)
{
    // FIXME: Read file into Buffer and pointers

    // FIXME: NeuronOrderInverse

    return true;
}

void CoactivationPredictor::Reset()
{
    Neighbors.clear();

    // Set all flags to false
    std::fill(SeenNeurons.begin(), SeenNeurons.end(), false);
}

int CoactivationPredictor::AddNeighbors(int neuron_index)
{
    int add_count = 0;

    if (SeenNeurons[neuron_index]) {
        return 0;
    }
    SeenNeurons[neuron_index] = true;

    const int above = NeighborsAbove[neuron_index];
    const int below = NeighborsBelow[neuron_index];

    for (int i = 1; i <= above; ++i) {
        const int neighbor_index = neuron_index + i;
        if (SeenNeurons[neighbor_index]) {
            continue;
        }
        SeenNeurons[neighbor_index] = true;

        Neighbors.push_back(static_cast<uint16_t>( neighbor_index ));
        ++add_count;
    }

    for (int i = 1; i <= below; ++i) {
        const int neighbor_index = neuron_index - i;
        if (SeenNeurons[neighbor_index]) {
            continue;
        }
        SeenNeurons[neighbor_index] = true;

        Neighbors.push_back(static_cast<uint16_t>( neighbor_index ));
        ++add_count;
    }

    // FIXME: Mega clusters here

    return add_count;
}
