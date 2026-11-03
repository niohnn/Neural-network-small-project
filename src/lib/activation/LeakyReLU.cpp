#include <activation/LeakyReLU.h>


dnn::LeakyReLU::LeakyReLU() {}

double dnn::LeakyReLU::activation(double x) {
    return x>0 ? x : 0.01*x;
}

double dnn::LeakyReLU::activationBackwards(double x) {
    return x>0 ? 1.0 : 0.01;
}

std::string dnn::LeakyReLU::getName() const {
    return "leaky_relu";
}

