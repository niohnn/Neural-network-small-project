#pragma once

#include <string>

namespace dnn {

class ActivationFunction {
public:
    virtual ~ActivationFunction()=default;

    virtual double activation(double x) = 0;
    virtual double activationBackwards(double x) = 0;
    virtual std::string getName() const = 0;
};

}
