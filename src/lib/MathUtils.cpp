#include <MathUtils.h>

#include <stdexcept>
#include <string>

namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b){
    double result = 0.0 ;
    for (std::size_t i = 0; i < a.size(); i++)
    {
        result += a[i]*b[i];
    }
    return result;
    
}

double sum(const std::vector<double>& values){
    double result = 0.0;
    for(double value: values){
        
       result+=value;
    }
    return result ;
}

std::vector<double> encodeAsOneHot(double value, std::size_t numClasses) {
    if (value < 0.0 || value >= static_cast<double>(numClasses)) {
        throw std::out_of_range("encodeAsOneHot: value " + std::to_string(value) + " is out of range for " +
                                std::to_string(numClasses) + " classes");
    }

    std::vector<double> encoded(numClasses, 0.0);
    encoded[static_cast<std::size_t>(value)] = 1.0;
    return encoded;
}

}
