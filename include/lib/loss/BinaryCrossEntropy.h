#pragma once

#include <loss/LossFunction.h>

namespace dnn {

class BinaryCrossEntropy : public LossFunction {
public:
    // Equivalent of get_cost_value/get_accuracy_value, adapted to this codebase's
    // vector-based (single-sample) design: there is no batch/m dimension, so the batch-axis
    // sums in the Python version collapse to a single sample's values.
    double getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override;
    double getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override;
    std::vector<double> getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const override;
    std::string getName() const override;
private:

    // Keeps predictedOutput away from exactly 0/1, where log() below would return -inf and the
    // gradient below would divide by zero.
    double clampProbability(double value) const {
        const double kProbabilityEpsilon = 1e-10;
        return std::min(std::max(value, kProbabilityEpsilon), 1.0 - kProbabilityEpsilon);
    }
};

}
