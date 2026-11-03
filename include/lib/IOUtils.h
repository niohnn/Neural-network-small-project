#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
namespace dnn {

class Network;

// Parses a CSV file where each line has labelSize label columns followed by inputSize feature
// columns (e.g. MNIST-style: labelSize=1, inputSize=784), appending each row's labelSize labels
// to labels and its inputSize features to inputs (as raw ints — see MathUtils::normalize to scale
// a row down to [0,1] doubles). Throws std::runtime_error if a row's column count doesn't
// equal labelSize + inputSize, or if the file can't be opened; inputs and labels are left
// unchanged in that case.
template <typename T>
void parseCsv(const std::string& filePath, std::vector<std::vector<T>>& inputs, std::size_t inputSize,
std::vector<std::vector<T>>& labels, std::size_t labelSize) {
        std::ifstream file(filePath);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + filePath);
        }

        // parse into locals so inputs/labels stay untouched if a row is malformed
        std::vector<std::vector<T>> parsedInputs;
        std::vector<std::vector<T>> parsedLabels;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }

            std::vector<std::string> fields;
            std::stringstream stream(line);
            std::string token;
            while (std::getline(stream, token, ',')) {
                fields.push_back(token);
            }

            if (fields.size() != labelSize + inputSize) {
                throw std::runtime_error("Row has " + std::to_string(fields.size()) + " columns, expected " +
                                         std::to_string(labelSize + inputSize) + " (labelSize=" +
                                         std::to_string(labelSize) + " + inputSize=" + std::to_string(inputSize) +
                                         "): " + filePath);
            }

            std::vector<T> label(labelSize);
            for (std::size_t i = 0; i < labelSize; ++i) {
                label[i] = std::stoi(fields[i]);
            }
            parsedLabels.push_back(std::move(label));

            std::vector<T> row(inputSize);
            for (std::size_t i = 0; i < inputSize; ++i) {
                row[i] = std::stoi(fields[labelSize + i]);
            }
            parsedInputs.push_back(std::move(row));
        }

        inputs = std::move(parsedInputs);
        labels = std::move(parsedLabels);
    }

// Dumps network to filePath as JSON-like text: the loss function's name, and per layer its
// activation function's name, weight matrix, and bias vector. Activation functions that need
// extra parameters (currently just ScaledSigmoid's min/max) also get "activationMin"/
// "activationMax" fields. Throws std::runtime_error if the file can't be opened for writing.
void saveModel(const Network& network, const std::string& filePath);

// Reads a file written by saveModel and replaces network's layers with the ones it describes
// (bypassing random initialization), reading back any "activationMin"/"activationMax" fields for
// activation functions that need them (currently just ScaledSigmoid). If network already has a
// loss function set, the file's saved loss-function name must match it, or this throws
// std::runtime_error; otherwise a matching LossFunction is created and set on network. Throws
// std::runtime_error if the file can't be opened or is malformed, or if it names an unknown
// activation function. network is left unchanged if an exception is thrown.
void loadModel(Network& network, const std::string& filePath);

}
