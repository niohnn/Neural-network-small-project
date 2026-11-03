#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

class ReLU: public ActivationFunction {
public:
    virtual ~ReLU()=default;
    ReLU();

    virtual double activation(double x);
    virtual double activationBackwards(double x);
    virtual std::string getName() const;
};

}
