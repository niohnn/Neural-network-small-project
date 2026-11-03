#pragma once

#include <activation/ActivationFunction.h>

#include <string>

// Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
class ScaledSigmoid {
public:
    ScaledSigmoid(double min, double max);

    double getMin() const;
    double getMax() const;

private:
    double min;
    double max;
};
