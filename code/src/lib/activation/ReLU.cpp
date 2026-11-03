#include <activation/ReLU.h>

namespace dnn {

double ReLU::activation(double x){
    return x > 0 ? x :0.0;
}

double ReLU::activationBackwards(double x) {
    return x > 0 ? 1.0 : 0.0;
}

std::string ReLU::getName() const {
    return "relu";
}

}