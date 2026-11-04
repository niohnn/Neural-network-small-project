#pragma once

#include <vector>
#include <algorithm>
namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);



// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);



template <typename T> //generic template declaration T can be int float double
std::vector<double>normalize(const std::vector<T>& values){
    T maxValue = *std::max_element(values.begin(),values.end());//normalize returns a vector containing double and accepts a vector of type T  const means cant modify ,values.begin points at first element values.end at last max element finds largest value
    std::vector<double>result; //creates an empty vector for normalized values 

    for(const T& value:values){
        result.push_back(static_cast<double>(value)/maxValue);//iterate through values then static_Cast convers value into a double then we divide by maxvalue then we push the result in the vector
    }
    return result;
}
template <typename T>
std::vector<double>normalize(const std::vector<T>& values,int maxValue){
    std::vector<double>result;
//create normalize  that returns vector of doubles but here input is generic type T and int max directly also we create new vector result 
    for(const T& value:values){
        result.push_back(static_cast<double>(value)/maxValue);
    } //same idea as the for loop previously 
    return result ;
}
                             

}
