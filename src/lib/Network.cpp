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

void Network ::fullForwardPropagation(const std::vector<double>& input){
    initialInput=input;
    std::vector<double> currentInput = input ;
    for(Layer* layer:layers){
        layer->forwardPropagation(currentInput);
        currentInput=layer->getActivationOutput();
    }
    output=currentInput;
}
//call forward propagation after saving initial input 
// currentInput is what we pass in the current layer ,network input values essentially 
// iteration through layers then with the foor loop 
//calculate current layer output with layer->
//pass output to the next layer with currentInput 
// store the final output in current input as outputs become inputs for the next neurones standard NN behaviour
void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput,
const std::vector<double>& actualOutput){
    if(ownedLossFunction == nullptr){
        throw std::logic_error("no loss function exists/is set");
    }
    std::vector<double>dACurr=
    ownedLossFunction->getGradient(predictedOutput,actualOutput);
    for (int i =static_cast<int>(layers.size())-1; i >=0; i--)
    {
    const std::vector<double>& previousLayerOutput=
    (i==0)? initialInput : layers[i-1]-> getPreActivationOutput();
    layers[i]->backwardPropagation(dACurr,previousLayerOutput);
    dACurr=layers[i]-> getDPreviousLayerOutput();
    }
    //dACurr is the gradient from the next layer 
    //previousLayerOutput is the activation output that was fed into this layer
    //backwardpropagation calculates the gradients and stored them in dPreviousLayerOutput
}

void Network ::trainBatch(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& actualOutputs,
    std::size_t batchSize,
    int epochs,
    double learningRate
){
    if(ownedLossFunction ==nullptr){
        throw std::logic_error("no loss function has been set yet");
    }
    if(inputs.empty() || actualOutputs.empty()){
        throw std::invalid_argument("inputs and outputs cant be empty");
    }
    if(inputs.size() != actualOutputs.size()){
        throw std::invalid_argument("input and actual output must be the same size");
    }
    if(batchSize == 0){
        throw std::invalid_argument("barch size needs to be larger than 0");

    }
    if(layers.empty()){
        throw std::invalid_argument("network needs at least 1 layer");
    }
    std::size_t eInputSize = layers.front()->getWeights().front().size();
    std::size_t eOutputSize = layers.back()->getBias().size();
    for(std::size_t i = 0; i<inputs.size();i++){
        if(inputs[i].size()!=eInputSize){
            throw std::invalid_argument("input size doesnt match network input size");
        }
        if(actualOutputs[i].size() != eOutputSize){
            throw std::invalid_argument("actual output size does not match the network output size");
        }
    }
    costHistory.clear();
    accuracyHistory.clear();
    for(int epoch=0; epoch<epochs; epoch++){
        double epochCost=0.0;
        double epochAccuracy= 0.0;
        for(std::size_t i =0; i < inputs.size();i++){
            fullForwardPropagation(inputs[i]);
            epochCost+=ownedLossFunction->getCostValue(output,actualOutputs[i]);
            epochAccuracy+=ownedLossFunction->getAccuracyValue(output,actualOutputs[i]);
            fullBackwardPropagation(output,actualOutputs[i]);
            if((i+1)%batchSize==0){
                for(Layer*layer:layers){
                    layer->commitToWeights(learningRate);
        
                }
            }

        }
        //comit any remaining gradientsfrom the final,incomplete batch
        if(inputs.size()%batchSize!=0){
            for(Layer* layer:layers){
                layer->commitToWeights(learningRate);
            }
        }
        epochCost /= inputs.size();
        epochAccuracy /= inputs.size();
        costHistory.push_back(epochCost);
        accuracyHistory.push_back(epochAccuracy);
    }
}
double Network::computeAccuracy(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& actualOutputs
) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("No loss function has been set");
    }

    if (inputs.empty() || actualOutputs.empty()) {
        throw std::invalid_argument("Inputs and actual outputs must not be empty");
    }

    if (inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Inputs and actual outputs must have the same size");
    }

    if (layers.empty()) {
        throw std::invalid_argument("Network must contain at least one layer");
    }

    std::size_t expectedInputSize = layers.front()->getWeights().front().size();
    std::size_t expectedOutputSize = layers.back()->getBias().size();

    double totalCorrect = 0.0;

    for (std::size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].size() != expectedInputSize) {
            throw std::invalid_argument("Input size does not match network input size");
        }

        if (actualOutputs[i].size() != expectedOutputSize) {
            throw std::invalid_argument("Actual output size does not match network output size");
        }

        fullForwardPropagation(inputs[i]);

        totalCorrect += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
    }

    return totalCorrect / inputs.size();
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

