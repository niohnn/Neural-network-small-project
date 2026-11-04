#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {

Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters, LossFunction*  lossFunction) 
: ownedLossFunction(lossFunction) {
    this->layers = std::vector<Layer*>(layerParameters.size());
    for (int i=0; i<layerParameters.size(); i++) {
        std::vector<std::vector<double>> weights = std::get<0>(layerParameters[i]);
        std::vector<double> bias = std::get<1>(layerParameters[i]);
        ActivationFunction* actFn = std::get<2>(layerParameters[i]);
        // use seed 0. parameters are overwritten in the next line
        // since the input layer of a network is not a Layer instance and "input as activation for layer 0", the layer id starts at 1
        Layer* layer = new Layer((uint64_t)(i+1), 0, weights.size()>0 ? weights[0].size() : 0, weights.size(), actFn);
        layer->setParameters(weights, bias);
        layers[i] = layer;
    }
}

Network::~Network() {
    delete ownedLossFunction;
    for (Layer* layer : layers) {
        delete layer;
    }
}

Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
    Layer* layer = new Layer(layers.size()+1, nextSeed, inputDim, outputDim, activationFunction);
    nextSeed++;
    layers.push_back(layer);
    return *this;
}

const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
    return layerId<=1 ? initialInput : layers[layerId-1]->getActivationOutput();
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    initialInput = input; // probably supposed to be used
    for (int i=0;i<layers.size();i++) {
        layers[i]->forwardPropagation(getLayerInput(i+1));
    }
    output = layers[layers.size()-1]->getActivationOutput();
}

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) {
    std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);

    for (int i=layers.size()-1;i>=0;i--) {
        layers[i]->backwardPropagation(dACurr, getLayerInput(i));
        dACurr = layers[i]->getDPreviousLayerOutput();
    }
}

void Network::trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
int epochs, double learningRate) {
    if (ownedLossFunction == NULL) throw std::invalid_argument("Loss function must be registered before trainBatch can be called.");
    if (!(inputs.size()==actualOutputs.size() && inputs.size()==batchSize && batchSize >0)) 
        throw std::invalid_argument("First dimension of inputs and actual outputs must match batch size, and batch size must be >0: Inputs first dimension = " + std::to_string(inputs.size()) 
        + ". Outputs first dimension = " + std::to_string(actualOutputs.size()) + ". Batch size" + std::to_string(batchSize));
    if (inputs[0].size() != layers[0]->getWeights()[0].size())
        throw std::invalid_argument("Input dimension doesn't match first layer's dimension");
    if (inputs[0].size() != layers[layers.size()-1]->getWeights().size())
        throw std::invalid_argument("Output dimension doesn't match last layer's dimension");
   
    costHistory.clear();
    accuracyHistory.clear();

    for (int i=0;i<batchSize;i++) {
        fullForwardPropagation(inputs[i]);
        
        double epochCost = ownedLossFunction->getCostValue(output, actualOutputs[i]);
        double epochAccuracy = ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);

        fullBackwardPropagation(output, actualOutputs[i]);
        if (i%batchSize==0) {
            for (int li=0; li<layers.size();li++) {
                layers[li]->commitToWeights(learningRate);
            }
        }

        costHistory.push_back(epochCost);
        accuracyHistory.push_back(epochAccuracy);
    }

}

double Network::computeAccuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs) {
    if (ownedLossFunction == NULL) throw std::invalid_argument("Loss function must be registered before trainBatch can be called.");
    if (!(inputs.size()==actualOutputs.size() && inputs.size() >0)) 
        throw std::invalid_argument("First dimension of inputs and actual outputs must match and be >0: Inputs first dimension = " + std::to_string(inputs.size()) 
        + ". Outputs first dimension = " + std::to_string(actualOutputs.size()));
    if (inputs[0].size() != layers[0]->getWeights()[0].size())
        throw std::invalid_argument("Input dimension doesn't match first layer's dimension");
    if (inputs[0].size() != layers[layers.size()-1]->getWeights().size())
        throw std::invalid_argument("Output dimension doesn't match last layer's dimension");

    double accuracy = 0;
    for (int i=0;i<inputs.size();i++) {
        fullForwardPropagation(inputs[i]);
        accuracy += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
    }
    return accuracy / (double)inputs.size();
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
