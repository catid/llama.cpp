#pragma once

#include <cstdint>
#include <vector>
#include <string>

static const uint32_t kCoActivationFileMagic = 0xc0ac7147;

/*
    This tool predicts what set of neurons will fire given a set of neurons
    that have been observed firing, based on the output of `study_co_activations`.

    We're specifically talking about the output of the non-linear activation
    and gating in a large language model, so the input to the output projection
    of the feed forward network in each block of the transformer model.

    For example, if we have observed that neuron #3 has fired, what other
    neurons are frequently observed firing as well?  And this process might
    proceed recursively.
*/

class CoactivationPredictor
{
public:
    ~CoactivationPredictor();

    int NeuronCount = -1;

    // Unique re-order of the original model neurons into a sorted order such
    // that neurons that often fire together are neighbors.
    uint16_t* NeuronOrder = nullptr;

    // Inverse of NeuronOrder
    std::vector<uint16_t> NeuronOrderInverse;

    // Number of correlated neighbors ahead (in index order after sorting).
    uint8_t* NeighborsAbove = nullptr;

    // Number of correlated neighbors behind (in index order after sorting).
    uint8_t* NeighborsBelow = nullptr;

    // Mega-cluster membership for each neuron.
    uint8_t* MegaCluster = nullptr;

    // Load the .coact file produced by the `study_co_activations` application.
    // On `calculate_inverse = true` this will fill NeuronOrderInverse.
    bool ReadFile(const std::string& file_path, bool calculate_inverse = true);

    /*
        Example usage:

        Reset()

            The Reset() function will clear internal state.

        For each neuron that fires in the initial set:
            AddNeighbors(neuron_index)

            This should be the sorted index given by NeuronOrder[].
            If you have not yet sorted your model weights,
            pass in neuron_index = NeuronOrder[original_neuron_index].

        The list of neurons that are neighbors is stored in the `Neighbors` member.
        It is fine to call Neighbors.clear() between calls to AddNeighbors().
    */
    void Reset();

    // Returns the number of neurons added to the Neighbors list.
    int AddNeighbors(int neuron_index);

    // Result of AddNeighbors.  Cleared on Reset().
    // If you have not sorted the neurons yet, you can convert this back to
    // the model neuron index via NeuronOrderInverse[neuron_index].
    std::vector<uint16_t> Neighbors;

protected:
    uint8_t* Buffer = nullptr;

    // Has a neuron been seen?  Either provided by AddNeighbors() or returned from it.
    // This is reset on Reset().
    std::vector<bool> SeenNeurons;
};
