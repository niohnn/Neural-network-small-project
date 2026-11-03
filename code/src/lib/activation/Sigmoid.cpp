#include <activation/Sigmoid.h>
#include <cmath>

namespace dnn {

double Sigmoid::activation(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::activationBackwards(double x) {
    double s = activation(x);
    return s * (1.0 - s);
}

std::string Sigmoid::getName() const {
    return "sigmoid";
}

}