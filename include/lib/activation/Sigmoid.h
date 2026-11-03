#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

class Sigmoid: public ActivationFunction {
public:
    virtual ~Sigmoid()=default;
    Sigmoid();

    virtual double activation(double x);
    virtual double activationBackwards(double x);
    virtual std::string getName() const;
};

}
