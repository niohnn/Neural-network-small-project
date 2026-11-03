#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn{

// Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
class ScaledSigmoid : public ActivationFunction {
public:
    ScaledSigmoid(double min, double max);

    double activation (double x) override ;
    double activationBackwards (double x) override ;
    std:: string getName() const override;

    double getMin() const;
    double getMax() const;

private:
    double min;
    double max;
};
}
