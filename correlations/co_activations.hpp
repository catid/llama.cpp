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

    // Number of correlated neighbors ahead (in index order after sorting).
    uint8_t* NeighborsAbove = nullptr;

    // Number of correlated neighbors behind (in index order after sorting).
    uint8_t* NeighborsBelow = nullptr;

    // Mega-cluster membership for each neuron.
    uint8_t* MegaCluster = nullptr;

    // Load the .coact file produced by the `study_co_activations` application.
    bool ReadFile(const std::string& file_path);

protected:
    uint8_t* Buffer = nullptr;
};
