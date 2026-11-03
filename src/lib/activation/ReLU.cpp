#include <activation/ReLU.h>


dnn::ReLU::ReLU() {}

double dnn::ReLU::activation(double x) {
    return x>0 ? x : 0.0;
}

double dnn::ReLU::activationBackwards(double x) {
    return x>0 ? 1.0 : 0.0;
}

std::string dnn::ReLU::getName() const {
    return "relu";
}

