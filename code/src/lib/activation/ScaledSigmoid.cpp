#include <activation/ScaledSigmoid.h>
#include <cmath>
namespace dnn{
ScaledSigmoid::ScaledSigmoid(double min,double max):min(min),max(max){

}
double ScaledSigmoid::activation(double x){
    return min+(max-min)/(1.0+std::exp(-x));
}
double ScaledSigmoid::activationBackwards(double x){
    double a = 1.0/(1.0 + std::exp(-x));
    return (max-min)*a*(1.0-a);
}
std::string ScaledSigmoid::getName() const{
    return "scaled_sigmoid";
}
}

