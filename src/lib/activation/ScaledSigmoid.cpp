#include <cmath>
#include <activation/ScaledSigmoid.h>


dnn::ScaledSigmoid::ScaledSigmoid(double min, double max) : min(min), max(max) {}

double dnn::ScaledSigmoid::getMin() const { return min; }
double dnn::ScaledSigmoid::getMax() const { return max; }

double dnn::ScaledSigmoid::activation(double x) {
    return min + (max-min) / (1+ std::exp(-x));
}

double dnn::ScaledSigmoid::activationBackwards(double x) {
    double sigm_fwd = 1/(1+std::exp(-x));
    return (max-min)*sigm_fwd*(1-sigm_fwd);
}

std::string dnn::ScaledSigmoid::getName() const {
    return "scaled_sigmoid";
}
