# Neuron Correlation Experiment

To enable the experimental features, build with -DLLAMA_CORRELATIONS=ON:

```bash
git clone https://github.com/catid/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake -DLLAMA_CORRELATIONS=ON ..
make -j16
```

You only have to manually build the software on the master node (see below).


## What is it?

This leverages the fast inference speed and hackable codebase of the `llama.cpp` project to collect neuron activation correlations.  This is a large triangular matrix of histogram bins that records which neurons in a large (7B parameter) model fire together given a set of example input data.

These correlation statistics are interesting data for projects related to speeding up inference of LLMs.  See the `Why` section at the end for why that is the case.


## How to run

To collect activations using a cluster of computers with GPUs with varying capabilties, first write a `servers.txt` file that looks like this:

```
gpu1.lan 15
gpu2.lan 15
gpu3.lan 20
gpu4.lan 20
gpu5.lan 20
gpu6.lan 20
```

The first column is the hostname of the computer, and the second column is the number of threads to use on the computer.  Machines with more threads will receive more work to do.

On each computer, you should check out this codebase at the same location on disk e.g. under `/home/catid/sources/llama.cpp/` using git.

Pick one computer to be your master node, and copy the model files under `./models/` just on this computer:

```bash
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
cd ..
```

On the master node, also download some test data:

```bash
mkdir -p data
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
unzip wikitext-2-raw-v1.zip?ref=salesforce-research
cd ..
```

Verify that the model runs by running:

```bash
./build/bin/perplexity -m models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q8_0.gguf -f data/wikitext-2-raw/wiki.test.raw
```

Important: Note the number of chunks reported on this line:

```
perplexity: calculating perplexity over 642 chunks, batch_size=512
```

You will have to provide the number e.g. `642` in the next step.

Finally, launch the correlation calculation job by running:

```bash
python scripts/launch_correlations.py 642
```

This will copy the models/data to the other computers, build the software on all the other computers, and compute the neuron activation correlations.  Before terminating each one will write the big correlation matrices to disk as `correlations_block_XX.zstd` - This is a Zstd compressed file with the format:

```
<Height of Matrix(UINT32)>
<First Row(UINT32)>
<Second Row(UINT32)> <Second Row(UINT32)>
<Third Row(UINT32)> <Third Row(UINT32)> <Third Row(UINT32)>
...
```

These files must be summed together to produce a single matrix for each block.  This is accomplished by running:

```bash
python scripts/collect_correlations.py
```

This script will create a `workspace` and `outputs` folders.  Files from other machines from `servers.txt` are copied into the `workspace`, and added to the `outputs` files using the built `./build/bin/sum_correlations` C++ application for speed.

Finally you can produce pretty correlation graphs by running:

```bash
./build/bin/analyze_correlations
```

This reads the files from `outputs` and generates `.png` heatmaps to visualize the full data, and provides interesting aggregated statistics about the data that can be graphed using other tools.


## Why is this useful?

The following papers have given me the strength and courage to dive deeper into studying neuron activation correlations.  Their research results are all pointing towards neuron activations being highly sparse, predictable, and correlated.  There are a lot of surprising findings, such as how sparsifying the activations actually improves the performance of LLMs on real-world benchmarks rather than hurting the model performance.

I suspect that we have yet to unlock all the best performance improvements possible by exploiting structure present in large pre-trained models.  And I believe these performance improvements will not require re-training the models, so independent researchers like myself may be able to contribute something useful.

I highly recommend digging into these, ranked in order of most interesting to least:


### Deja Vu

Paper: https://arxiv.org/pdf/2310.17157.pdf

Key take-aways from Deja Vu paper:
(1) In the first pass, we record a subset of parameters, specifically which attention heads and MLP neurons yield large output norms for the input. In the second pass, each input example only uses the recorded subset of parameters for the computation. Surprisingly, these two forward passes lead to similar prediction or performance on all in-context learning and language modeling tasks.
TL;DR: You can chop off weakly-activating neurons and still do well on downstream tasks!
(2) Attention heads seem to perform a sort of "mean-shift step" that may explain why token embeddings tend to cluster after going through several layers.
(3) The similarity of embeddings between blocks  is high!  About 99% of the embeddings at the input of one block will still exist at the output of that block.  They postulate this has to do with the residual connection.  This means that each block is mostly calculating a 1% change to the embedding.
(4) They train a small MLP to predict which group of MLP neurons will fire when given an input.  This exploits neuron activation correlation similar to the MoEfication paper.
(5) The authors also find that downstream accuracy improves in some benchmarks with more sparsity.
(6) Sparse activation is entirely compatible with quantization (they checked).
(7) You can also sparsify the transformer blocks with similar good results (and about the same amount of sparsity) 


### MoEfication

Code: https://github.com/thunlp/MoEfication

Fascinating paper that uses activation correlations to demonstrate that neurons fire in correlated clusters.  They leverage this to MoE-ify models without retraining the models, and get really good results!

This leads to a 2x speedup in inference with minimal quality loss.
Follow-on work here further studies this emergent behavior in large networks: https://arxiv.org/pdf/2305.18390.pdf
This actually works so well it's incorporated into an optimization package now: https://github.com/OpenBMB/BMCook

This paper is the reason I am investing a lot of time into studying neuron activation correlations.


### Toy Models

Article: https://transformer-circuits.pub/2022/toy_model/index.html

It provides a good explanation for grokking
It explains that neuron activations in a well trained LLM are probably hard to predict from the input features because neurons correspond to multiple features in superposition
It doesn’t explain why MoE works
But they give me some hope that simple operations can approximate the action of the router in MoE
Because activations are limited to just a few input features, and the most important features often are closely related to activations, and the first level of super position can be pulled apart with a binary operation
Also if superposition works it is because the inputs are sparse and interpretable

So perhaps the most important neurons are the easiest to predict, and from those I can span the graph rapidly
They also note that MoE and transformer neurons are easier to predict but don’t go into detail


### PowerInfer

Paper: https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf

Key take-aways from PowerInfer paper:
(1) neuron activation in an LLM follows a skewed power-law distribution: a small subset of neurons consistently contributes to the majority of activations (over 80%)
(2) they're mainly concerned with CPU+GPU hybrid setups where you have to keep deleting and copying weights from CPU to GPU for each layer, so they're optimizing for PCIe bandwidth
(3) it does not exploit neuron activation correlation so seems like it could be greatly improved
(4) pruning activations seems to improve accuracy in benchmarks in some cases


### LLM in a flash

Paper: https://arxiv.org/pdf/2312.11514.pdf

Key take-aways from this new paper from Apple:
(0) They're mainly optimizing for memory usage, obviously concerned that LLMs don't fit into the tiny RAM on their cellphones.
(1) Based heavily on the Deja Vu paper results.
(2) Avoids loading the whole model in system RAM by loading a moving window of parameters.  Stacks up/down rows/columns in the FFNs on disk to read them faster.  They claim it works for models up to 2x as large as system RAM, which means pretty much any model can run on a desktop PC.  Uses multiple parallel IO reads of ~32KB.
(3) They only load 2% of FFN parameters from flash for each query.
(4) Like all existing work so far they only consider models using ReLU. - I think SILU/GELU can also work if you include +/- activations.
(5) They tried to use correlated activations but stopped short.  They found that some neurons activate almost all the time and gave up and used the Deja Vu approach.  This is interesting though because it means some neurons should just always be evaluated.  I think they missed a big opportunity here.
