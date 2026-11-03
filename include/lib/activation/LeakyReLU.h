#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

class LeakyReLU: public ActivationFunction {
public:
    virtual ~LeakyReLU()=default;
    LeakyReLU();

    virtual double activation(double x);
    virtual double activationBackwards(double x);
    virtual std::string getName() const;
};

}
