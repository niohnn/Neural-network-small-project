#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {


class ReLU : public ActivationFunction {
    public :
    ReLU()= default;
    ~ReLU () override = default;

    double activation (double x) override ;
    double activationBackwards(double x ) override ;
    std::string getName() const override;
};

}
