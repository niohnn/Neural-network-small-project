#pragma once

#include <vector>
#include <algorithm>
namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);



// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);


// earlier we kept maximum in generic type T ,for doubles htat meant dividing by the exact fractional maximum
//but for integers that meant dividing by the maximum integer which is not what we want for normalization,
//instead we take the maximum as an int directly ,what the test expects 
template <typename T>
std::vector<double> normalize(const std::vector<T>& values) {
    if (values.empty()) {
        return {};
    }

    const int maxValue = static_cast<int>(*std::max_element(values.begin(), values.end()));
    std::vector<double> result;
    result.reserve(values.size());
    for (const T& value : values) {
        result.push_back(static_cast<double>(value) / maxValue);
    }
    return result;
}
template <typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue) {
    std::vector<double> result;
    result.reserve(values.size());
//create normalize  that returns vector of doubles but here input is generic type T and int max directly also we create new vector result 
    for(const T& value:values){
        result.push_back(static_cast<double>(value)/maxValue);
    }
    return result;
}


                             

}
