#pragma once

#include <string>
#include <vector>

namespace dnn {

class LossFunction {
public:
    virtual ~LossFunction() = default;

    virtual double getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const = 0;
    virtual double getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const = 0;

    // dLoss/dPredictedOutput for each output unit, given one sample's predicted and actual
    // output. Seeds Network::fullBackwardPropagation()'s backward pass, so each LossFunction
    // owns its own gradient formula (and any numerical safeguards it needs, e.g. clamping away
    // from a division-by-zero point) instead of Network hardcoding one loss's math.
    virtual std::vector<double> getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const = 0;

    virtual std::string getName() const = 0;
};

}
