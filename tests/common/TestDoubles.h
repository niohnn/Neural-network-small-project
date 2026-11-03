#pragma once

// Activation and loss functions used by the Layer and Network tests instead of the real ones, so
// those tests neither depend on the student's activation/loss implementations nor fail to compile
// while the activation headers are still unfinished. The reference bridge compiles this header with
// dnn renamed to dnn_ref, so both sides use identical doubles.

#include <activation/ActivationFunction.h>
#include <loss/LossFunction.h>

#include <cmath>
#include <string>
#include <vector>

namespace dnn {

class TestTanh : public ActivationFunction {
public:
    double activation(double x) override { return std::tanh(x); }
    double activationBackwards(double x) override {
        const double t = std::tanh(x);
        return 1.0 - t * t;
    }
    std::string getName() const override { return "test_tanh"; }
};

// Asymmetric slopes and an offset, so that mixing up activation and derivative, or Z and A, shows.
class TestPiecewiseLinear : public ActivationFunction {
public:
    double activation(double x) override { return x > 0.0 ? 2.0 * x + 0.1 : 0.5 * x + 0.1; }
    double activationBackwards(double x) override { return x > 0.0 ? 2.0 : 0.5; }
    std::string getName() const override { return "test_piecewise_linear"; }
};

class TestSquaredError : public LossFunction {
public:
    double getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override {
        double cost = 0.0;
        for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
            cost += (predictedOutput[i] - actualOutput[i]) * (predictedOutput[i] - actualOutput[i]);
        }
        return cost / static_cast<double>(predictedOutput.size());
    }
    double getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override {
        std::size_t correct = 0;
        for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
            if (std::abs(predictedOutput[i] - actualOutput[i]) < 0.5) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(predictedOutput.size());
    }
    std::vector<double> getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override {
        std::vector<double> gradient(predictedOutput.size());
        for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
            gradient[i] = 2.0 * (predictedOutput[i] - actualOutput[i]) / static_cast<double>(predictedOutput.size());
        }
        return gradient;
    }
    std::string getName() const override { return "test_squared_error"; }
};

}
