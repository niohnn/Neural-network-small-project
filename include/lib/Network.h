#pragma once

#include <Layer.h>
#include <activation/ActivationFunction.h>
#include <loss/LossFunction.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace dnn {

class Network {
public:

    Network()=default;
    ~Network();

    // Not copyable, since a copy would delete the layers and loss function a second time.
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    // Builds a Network directly from pretrained per-layer parameters, bypassing
    // architecture-string parsing and random initialization. layerParameters holds one
    // (weights, bias, activationFunction) tuple per layer, in order; each layer's output size
    // is the number of rows in its weight matrix (and the length of its bias vector), and its
    // input size is the number of columns. Takes ownership of the activation functions and lossFunction.
    Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters,
            LossFunction*  lossFunction);

    // Appends a new, freshly-initialized layer (inputDim -> outputDim) using activationFunction,
    // and returns this Network so calls can be chained into a fluent setup, e.g.:
    //   network.addLayer(2, 4, new ReLU()).addLayer(4, 1, new Sigmoid());
    // Takes ownership of activationFunction.
    Network& addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction);

    // Equivalent of full_forward_propagation. Stores the network input and A_curr; each
    // layer's own input (A_prev) and pre-activation output (Z_curr) are available on demand
    // via getLayerInput() and Layer::getPreActivationOutput(). See the getters below.
    void fullForwardPropagation(const std::vector<double>& input);

    // Equivalent of full_backward_propagation, adapted to this vector-based (single-sample)
    // design: predictedOutput/actualOutput take the place of Y_hat/Y for a single sample, and
    // there is no grads_values map, since each layer's own gradients (dW_curr/db_curr) are
    // stored on that Layer by Layer::backwardPropagation(); query them via
    // Layer::getDWeights() / Layer::getDBias() after this call. Seeds the backward pass via the
    // loss function set via setLossFunction()'s getGradient(); throws std::logic_error if none has been set.
    void fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput);

    // For each epoch, runs a forward+backward pass (via fullForwardPropagation()/
    // fullBackwardPropagation()) over every sample in inputs; each layer accumulates its own
    // gradients internally (see Layer::backwardPropagation()) instead of applying them
    // immediately, and a Layer::commitToWeights() call is made on every layer after every
    // batchSize samples (plus once more at the end of the epoch to flush a final, possibly
    // smaller batch), applying the gradients averaged over that batch. This reduces the
    // per-sample gradient noise that plain single-sample SGD is prone to. costHistory/
    // accuracyHistory store one whole-dataset-averaged value per epoch (cleared and
    // repopulated by each call). Uses the loss function set via setLossFunction(); throws
    // std::logic_error if none has been set, and std::invalid_argument if inputs/actualOutputs
    // mismatch in size or are empty, or if batchSize is 0.
    void trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
                     int epochs, double learningRate);

    // Runs a forward pass (via fullForwardPropagation()) over each of inputs and compares it
    // against the matching entry in actualOutputs using the loss function's
    // getAccuracyValue(), returning the fraction of samples classified correctly (in [0,1]).
    // Not const: like training, each forward pass overwrites this Network's own
    // getOutput()/getLayerInput() state. Uses the loss function set via setLossFunction();
    // throws std::logic_error if none has been set, and std::invalid_argument if
    // inputs/actualOutputs mismatch in size or are empty.
    double computeAccuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs);

    // Takes ownership of lossFunction and deletes the previously-set one.
    void setLossFunction(LossFunction* lossFunction);
    const LossFunction* getLossFunction() const;

    const std::vector<double>& getOutput() const;

    // A_prev of the layer with the given id: the network input for id 0, otherwise the
    // activation output of the preceding layer.
    const std::vector<double>& getLayerInput(uint64_t layerId) const;

    const std::vector<double>& getCostHistory() const;
    const std::vector<double>& getAccuracyHistory() const;

    // All of this Network's layers, in order (used by IOUtil::saveModel).
    const std::vector<Layer*>& getLayers() const;

    // Replaces this Network's layers wholesale (used by IOUtil::loadModel), resetting the
    // dependent per-forward-pass state (network input/output, cost/accuracy history), since it
    // no longer corresponds to the new layers. Takes ownership of newLayers and deletes the old layers.
    void setLayers(std::vector<Layer*> newLayers);

private:
    std::vector<Layer*> layers;

    LossFunction* ownedLossFunction = nullptr;

    uint64_t nextSeed = 42;

    std::vector<double> initialInput;
    std::vector<double> output;

    std::vector<double> costHistory;
    std::vector<double> accuracyHistory;
};

}
