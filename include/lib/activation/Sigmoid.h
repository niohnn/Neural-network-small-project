#pragma once

#include <activation/ActivationFunction.h>

#include <string>
namespace dnn {

class Sigmoid :public ActivationFunction {
    public :
    Sigmoid()=default;
    ~Sigmoid() override = default;
    double activation(double x) override ;
    double activationBackwards(double x) override ;
    std::string getName() const override;
};
}
