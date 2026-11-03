#include <cmath>
#include <activation/Sigmoid.h>


dnn::Sigmoid::Sigmoid() {}

double dnn::Sigmoid::activation(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double dnn::Sigmoid::activationBackwards(double x) {
    const double act = activation(x);
    return act * (1-act);
}

std::string dnn::Sigmoid::getName() const {
    return "sigmoid";
}

