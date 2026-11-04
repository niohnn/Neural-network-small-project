#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}

void Layer::forwardPropagation(const std::vector<double>& previousLayerOutput) {
    std::vector<double> preActivationOutput(layerOutputSize, 0.0);
    std::vector<double> activationOutput(layerOutputSize, 0.0);
    const std::vector<std::vector<double>>& weights = getWeights();
    const std::vector<double>& bias = getBias();
    for (int i=0; i<layerOutputSize;i++) {
        preActivationOutput[i] = dotProduct(weights[i], previousLayerOutput) + bias[i];
        activationOutput[i] = activationFunction->activation(preActivationOutput[i]);
    }
}

void Layer::backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput) {
    std::vector<double> dPreActivation(layerOutputSize, 0.0);
    for (int i=0; i<layerOutputSize;i++) {
        dPreActivation[i] = dActivationOutput[i] * activationFunction->activationBackwards(preActivationOutput[i]);
    }
    std::vector<std::vector<double>> dWeights(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    for (int i=0; i<layerOutputSize;i++) {
        for (int j=0;j<layerInputSize;j++) {
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    }
    dBias = std::move(dPreActivation);
    for (int i=0; i<accumulatedDWeights.size();i++) {
        accumulatedDWeights[i] = vecAdd(accumulatedDWeights[i], dWeights[i]);
    }
    accumulatedDBias = vecAdd(accumulatedDBias, dBias);
    numProcessedSamples++;
    std::vector<double> dPreviousLayerOutput(layerInputSize, 0.0);
    for (int j=0; j<layerInputSize; j++) {
        std::vector<double> wCol = getMatrixColumn(weights, j);
        dPreviousLayerOutput[j] = dotProduct(wCol, dPreActivation);
    }
}

void Layer::updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate) {
    for (int i=0; i<weights.size(); i++) {
        std::vector<double> difference = vecScalarMul(dWeights[i], -learningRate);
        weights[i] = vecAdd(weights[i], difference);
    }
    std::vector<double> difference = vecScalarMul(dBias, -learningRate);
    bias = vecAdd(bias, difference);
}

void Layer::commitToWeights(double learningRate) {
    for (int i=0; i<weights.size(); i++) {
        accumulatedDWeights[i] = vecScalarMul(accumulatedDWeights[i], 1/(double)numProcessedSamples);
    }
    accumulatedDBias = vecScalarMul(accumulatedDBias, 1/(double)numProcessedSamples);
    updateParameters(accumulatedDWeights, accumulatedDBias, learningRate);

    for (int i=0; i<layerOutputSize; i++) {
        for (int j=0; j<layerInputSize; j++) {
            accumulatedDWeights[i][j] = 0.0;
        }
        accumulatedDBias[i] = 0.0;
    }
    numProcessedSamples=0;
}

uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}

}
