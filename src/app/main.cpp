#include <Network.h>
#include <loss/BinaryCrossEntropy.h>

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "IOUtils.h"
#include "MathUtils.h"
#include "activation/ReLU.h"
#include "activation/Sigmoid.h"


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generateSamples(
    int count, double start, double end,
    const std::function<bool(double, double)>& predicate = [](double x, double y) { return x + y > 1; }) {
    std::random_device randomDevice;
    std::mt19937 generator(100);
    std::uniform_real_distribution<double> distribution(start, end);

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    inputs.reserve(count);
    outputs.reserve(count);
    for (int i = 0; i < count; ++i) {
        const double x = distribution(generator);
        const double y = distribution(generator);
        inputs.push_back({x, y});
        outputs.push_back({predicate(x, y) ? 1.0 : 0.0});
    }
    return {inputs, outputs};
}

int main() {
    try {
        //const auto [inputs, outputs] = generateSamples(10000, 0.0, 1.0);
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> outputs;
        dnn::parseCsv("../mnist_test.csv",inputs,784,outputs,1);

        for (auto & input : inputs) {
            input=dnn::normalize(input,255);
        }

        for (auto & output : outputs) {
            output = dnn::encodeAsOneHot(output[0], 10);
        }

        dnn::Network network;
        /*network.addLayer(2,4,new dnn::LeakyReLU()).
                addLayer(4,6,new dnn::LeakyReLU()).
                addLayer(6,6,new dnn::LeakyReLU()).
                addLayer(6,4,new dnn::LeakyReLU()).
                addLayer(4,1,new dnn::Sigmoid()).
                setLossFunction(new dnn::BinaryCrossEntropy());
        */
        network.addLayer(784,128,new dnn::ReLU()).
            addLayer(128,64,new dnn::ReLU()).
            addLayer(64,10,new dnn::Sigmoid()).
            setLossFunction(new dnn::BinaryCrossEntropy());

        const int epochs = 10;
        const double learningRate = 0.01;
        const std::size_t batchSize = 32;

        std::cout << "Training on " << inputs.size() << " samples in batches of up to " << batchSize << "...\n";

        network.trainBatch(inputs, outputs, batchSize, epochs, learningRate);

        for (std::size_t epoch = 0; epoch < network.getCostHistory().size(); ++epoch) {
            if (epoch == 0 || (epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << (epoch + 1) << ": cost=" << network.getCostHistory()[epoch]
                           << ", accuracy=" << network.getAccuracyHistory()[epoch] << "\n";
            }
        }

        std::cout << "Full-dataset accuracy after training: " << network.computeAccuracy(inputs, outputs) << "\n";

        /*network.fullForwardPropagation({0.1,0.1});
        auto output=network.getOutput();
        std::cout<<std::ceil(output.at(0) * 100.0) / 100.0<<"\n";

        network.fullForwardPropagation({0.8,0.5});
        output=network.getOutput();
        std::cout<<std::ceil(output.at(0) * 100.0) / 100.0<<"\n";
        */

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}
