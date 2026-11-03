#include <IOUtils.h>

#include <Network.h>
#include <activation/LeakyReLU.h>
#include <activation/ReLU.h>
#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>
#include <loss/BinaryCrossEntropy.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace dnn {
    ActivationFunction* createActivationFunction(const std::string& name, double activationMin, double activationMax) {
        if (name == "sigmoid") {
            return new Sigmoid();
        }
        if (name == "relu") {
            return new ReLU();
        }
        if (name == "leaky_relu") {
            return new LeakyReLU();
        }
        if (name == "scaled_sigmoid") {
            return new ScaledSigmoid(activationMin, activationMax);
        }
        return nullptr;
    }

    LossFunction* createLossFunction(const std::string& name) {
        if (name == "binary_cross_entropy") {
            return new BinaryCrossEntropy();
        }
        return nullptr;
    }

    void writeJsonArray(std::ofstream& file, const std::vector<double>& values) {
        file << "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                file << ", ";
            }
            file << values[i];
        }
        file << "]";
    }

    // Parses a flat JSON-like number array, e.g. "[1, -2.5, 3]".
    std::vector<double> parseNumberArray(const std::string& arrayText) {
        std::vector<double> values;
        const std::string inner = arrayText.substr(1, arrayText.size() - 2);

        std::stringstream stream(inner);
        std::string token;
        while (std::getline(stream, token, ',')) {
            values.push_back(std::stod(token));
        }
        return values;
    }

    void saveModel(const Network& network, const std::string& filePath) {
        std::ofstream file(filePath);
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + filePath);
        }

        file << std::setprecision(17);

        const auto& layers = network.getLayers();
        const LossFunction* lossFunction = network.getLossFunction();

        file << "{\n";
        file << "  \"lossFunction\": \"" << (lossFunction ? lossFunction->getName() : "") << "\",\n";
        file << "  \"layers\": [\n";

        for (std::size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
            const Layer& layer = *layers[layerIndex];
            const ActivationFunction* activationFunction = layer.getActivationFunction();

            file << "    {\n";
            file << "      \"activation\": \"" << (activationFunction ? activationFunction->getName() : "") << "\",\n";

            if (const auto* scaledSigmoid = dynamic_cast<const ScaledSigmoid*>(activationFunction)) {
                file << "      \"activationMin\": " << scaledSigmoid->getMin() << ",\n";
                file << "      \"activationMax\": " << scaledSigmoid->getMax() << ",\n";
            }

            file << "      \"weights\": [";
            const auto& weights = layer.getWeights();
            for (std::size_t i = 0; i < weights.size(); ++i) {
                if (i > 0) {
                    file << ", ";
                }
                writeJsonArray(file, weights[i]);
            }
            file << "],\n";

            file << "      \"bias\": ";
            writeJsonArray(file, layer.getBias());
            file << "\n";

            file << "    }" << (layerIndex + 1 < layers.size() ? "," : "") << "\n";
        }

        file << "  ]\n";
        file << "}\n";
    }

    void loadModel(Network& network, const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + filePath);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string content = buffer.str();

        static const std::regex lossFunctionPattern(R"~("lossFunction"\s*:\s*"([^"]*)")~");
        static const std::regex layerObjectPattern(R"(\{[^{}]*\})");
        static const std::regex activationPattern(R"~("activation"\s*:\s*"([^"]+)")~");
        static const std::regex activationMinPattern(R"~("activationMin"\s*:\s*([^,\s}]+))~");
        static const std::regex activationMaxPattern(R"~("activationMax"\s*:\s*([^,\s}]+))~");
        static const std::regex numberArrayPattern(R"(\[[^\[\]]*\])");

        std::smatch lossFunctionMatch;
        if (!std::regex_search(content, lossFunctionMatch, lossFunctionPattern)) {
            throw std::runtime_error("Model file missing lossFunction field: " + filePath);
        }
        const std::string savedLossFunctionName = lossFunctionMatch[1];

        // network is only modified once the whole file has been parsed successfully
        const LossFunction* lossFunction = network.getLossFunction();
        LossFunction* newLossFunction = nullptr;
        if (lossFunction) {
            if (lossFunction->getName() != savedLossFunctionName) {
                throw std::runtime_error("Model's loss function (\"" + savedLossFunctionName +
                    "\") does not match the currently-set loss function (\"" + lossFunction->getName() + "\")");
            }
        }
        else {
            newLossFunction = createLossFunction(savedLossFunctionName);
            if (!newLossFunction) {
                throw std::runtime_error("Unknown loss function in model file: " + savedLossFunctionName);
            }
        }

        std::vector<Layer*> newLayers;
        ActivationFunction* layerActivationFunction = nullptr;
        uint64_t seed = 42;

        // on a parse error, free everything allocated so far before rethrowing
        try {
            for (auto it = std::sregex_iterator(content.begin(), content.end(), layerObjectPattern); it !=
                 std::sregex_iterator(); ++it) {
                const std::string layerText = it->str();

                std::smatch activationMatch;
                if (!std::regex_search(layerText, activationMatch, activationPattern)) {
                    throw std::runtime_error("Layer missing activation field in model file: " + filePath);
                }
                double activationMin = 0.0;
                double activationMax = 1.0;
                std::smatch activationMinMatch;
                if (std::regex_search(layerText, activationMinMatch, activationMinPattern)) {
                    activationMin = std::stod(activationMinMatch[1]);
                }
                std::smatch activationMaxMatch;
                if (std::regex_search(layerText, activationMaxMatch, activationMaxPattern)) {
                    activationMax = std::stod(activationMaxMatch[1]);
                }

                layerActivationFunction = createActivationFunction(activationMatch[1], activationMin, activationMax);
                if (!layerActivationFunction) {
                    throw std::runtime_error("Unknown activation function in model file: " + activationMatch[1].str());
                }

                const std::size_t weightsKeyPos = layerText.find("\"weights\"");
                const std::size_t biasKeyPos = layerText.find("\"bias\"");
                if (weightsKeyPos == std::string::npos || biasKeyPos == std::string::npos || biasKeyPos < weightsKeyPos) {
                    throw std::runtime_error("Layer missing weights/bias fields in model file: " + filePath);
                }

                const std::string weightsSection = layerText.substr(weightsKeyPos, biasKeyPos - weightsKeyPos);
                const std::string biasSection = layerText.substr(biasKeyPos);

                std::vector<std::vector<double>> weights;
                for (auto rowIt = std::sregex_iterator(weightsSection.begin(), weightsSection.end(), numberArrayPattern);
                     rowIt != std::sregex_iterator(); ++rowIt) {
                    weights.push_back(parseNumberArray(rowIt->str()));
                }

                std::smatch biasArrayMatch;
                if (!std::regex_search(biasSection, biasArrayMatch, numberArrayPattern)) {
                    throw std::runtime_error("Layer missing bias array in model file: " + filePath);
                }
                std::vector<double> bias = parseNumberArray(biasArrayMatch.str());

                if (weights.empty() || weights.size() != bias.size()) {
                    throw std::runtime_error("Layer weights/bias size mismatch in model file: " + filePath);
                }

                const uint64_t layerOutputSize = weights.size();
                const uint64_t layerInputSize = weights[0].size();
                const uint64_t id = newLayers.size();

                Layer* layer = new Layer(id, seed++, layerInputSize, layerOutputSize, layerActivationFunction);
                layerActivationFunction = nullptr; // now owned by layer
                newLayers.push_back(layer);
                newLayers.back()->setParameters(std::move(weights), std::move(bias));
            }

            if (newLayers.empty()) {
                throw std::runtime_error("Model file contains no layers: " + filePath);
            }
        }
        catch (...) {
            delete layerActivationFunction;
            for (Layer* layer : newLayers) {
                delete layer;
            }
            delete newLossFunction;
            throw;
        }

        if (newLossFunction) {
            network.setLossFunction(newLossFunction);
        }
        network.setLayers(std::move(newLayers));
    }
}
