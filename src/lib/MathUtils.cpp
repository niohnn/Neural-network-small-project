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

std::vector<double> vecAdd(std::vector<double>& a, std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::exception("Vectors need to have same dimensions");
    }
    std::vector<double> c(a);
    for (int i=0;i<a.size(); i++) {
        c[i]+=b[i];
    }
    return c;
} 

std::vector<double> getMatrixColumn(std::vector<std::vector<double>>& matrix, size_t columnIndex) {
    if (matrix.size() == 0) {
        std::vector<double> emptyVec(0, 0.0);
        return emptyVec;
    }
    size_t dim = matrix[0].size();
    std::vector<double> column(dim, 0.0);
    for (int i=0; i<matrix.size(); i++) {
        if (matrix[i].size() != dim) throw std::exception("Inconsistent shape");
        column[i] = matrix[i][columnIndex];
    }
    return column;
}

std::vector<double> vecScalarMul(std::vector<double> a, double b) {
    std::vector c(a);
    for (int i=0;i<a.size();i++) {
        c[i] *= b;
    }
    return c;
}

}
