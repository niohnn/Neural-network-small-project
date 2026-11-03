#include <activation/LeakyReLU.h>
namespace dnn {
double LeakyReLU::activation(double x){
    return x >0 ? x : 0.01*x;
}
double LeakyReLU::activationBackwards(double x){
    return x >0 ? 1 :0.01;
}
std::string LeakyReLU::getName() const{
    return "leakyRelu";
}
}
