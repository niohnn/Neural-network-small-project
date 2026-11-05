#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {
   // Constructor
Network::Network(
    std::vector<std::tuple<
        std::vector<std::vector<double>>,
        std::vector<double>,
        ActivationFunction*
    >> layerParameters,
    LossFunction* lossFunction
) : ownedLossFunction(lossFunction) {

    for (std::size_t i = 0; i < layerParameters.size(); i++) {
        auto weights = std::get<0>(layerParameters[i]);
        auto bias = std::get<1>(layerParameters[i]);
        auto activationFunction = std::get<2>(layerParameters[i]);

        uint64_t inputSize = weights[0].size();
        uint64_t outputSize = weights.size();

        Layer* layer = new Layer(
            i + 1, 0, inputSize, outputSize, activationFunction
        );

        layer->setParameters(weights, bias);
        layers.push_back(layer);
    }
}

// Destructor
Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }

    delete ownedLossFunction;
} 
Network& Network::addLayer(
    uint64_t inputDim,
    uint64_t outputDim,
    ActivationFunction* activationFunction
) {
    Layer* layer = new Layer(
        layers.size() + 1,
        nextSeed,
        inputDim,
        outputDim,
        activationFunction
    );

    nextSeed++;
    layers.push_back(layer);

    return *this;
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}

