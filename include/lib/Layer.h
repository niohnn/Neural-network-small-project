#pragma once

#include <activation/ActivationFunction.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dnn {

class Layer {
public:
    // Takes ownership of activationFunction (deleted in ~Layer()).
    Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
          ActivationFunction* activationFunction, std::string  name = "");
    ~Layer();

    // Not copyable, since a copy would delete activationFunction a second time.
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    void init();

    // Directly sets weights/bias (e.g. when loading a pretrained model), bypassing init()'s
    // random initialization.
    void setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias);

    // Stores A_curr and Z_curr in activationOutput/preActivationOutput; see the getters below.
    void forwardPropagation(const std::vector<double>& previousLayerOutput);

    // Equivalent of single_layer_backward_propagation, adapted to this vector-based
    // (single-sample) design: there is no batch/m dimension, so the batch-axis sum and the
    // division by m in the Python version collapse to identity operations here. Stores
    // dA_prev/dW_curr/db_curr; see the getters below. Throws std::logic_error if the layer has no activation
    // function (the equivalent of the Python function's "non-supported activation" exception).
    void backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput);

    // Averages the gradients accumulated across calls to backwardPropagation() since the
    // last commitToWeights() (or construction), applies one gradient-descent step, then
    // resets the accumulators to zero for the next batch. No-op if backwardPropagation()
    // has not been called since the last commit.
    void commitToWeights(double learningRate);

    uint64_t getId() const;
    const std::string& getName() const;
    const std::vector<double>& getActivationOutput() const;
    const std::vector<double>& getPreActivationOutput() const;
    ActivationFunction* getActivationFunction() const;
    const std::vector<std::vector<double>>& getWeights() const;
    const std::vector<double>& getBias() const;

    const std::vector<double>& getDPreviousLayerOutput() const;
    const std::vector<std::vector<double>>& getDWeights() const;
    const std::vector<double>& getDBias() const;

private:
    uint64_t id;
    std::string name;
    uint64_t seed;
    uint64_t layerInputSize;
    uint64_t layerOutputSize;
    ActivationFunction* activationFunction;

    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    std::vector<double> activationOutput;
    std::vector<double> preActivationOutput;

    std::vector<double> dPreviousLayerOutput;
    std::vector<std::vector<double>> dWeights;
    std::vector<double> dBias;

    std::vector<std::vector<double>> accumulatedDWeights;
    std::vector<double> accumulatedDBias;
    uint64_t numProcessedSamples = 0;

    // Gradient-descent step using explicitly-supplied gradients (e.g. a mini-batch average).
    // Only called internally by commitToWeights().
    void updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate);
};

}
