// Checks the class declarations in the headers of the activation functions. Only the headers are
// compiled (nothing is called or linked), so missing or mismatching declarations show up as compiler
// errors naming the failed check.

#include <activation/ActivationFunction.h>
#include <activation/LeakyReLU.h>
#include <activation/ReLU.h>
#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace {

template <typename T>
constexpr bool isActivationFunction() {
    static_assert(std::is_base_of_v<dnn::ActivationFunction, T>, "must derive from dnn::ActivationFunction");
    static_assert(std::is_convertible_v<T*, dnn::ActivationFunction*>, "must derive publicly from dnn::ActivationFunction");
    static_assert(!std::is_abstract_v<T>, "must override activation, activationBackwards and getName");
    static_assert(std::is_same_v<decltype(&T::activation), double (T::*)(double)>, "must declare double activation(double)");
    static_assert(std::is_same_v<decltype(&T::activationBackwards), double (T::*)(double)>,
                  "must declare double activationBackwards(double)");
    static_assert(std::is_same_v<decltype(&T::getName), std::string (T::*)() const>, "must declare std::string getName() const");
    return true;
}

}

TEST(ActivationHeaders, Sigmoid) {
    static_assert(isActivationFunction<dnn::Sigmoid>());
    static_assert(std::is_default_constructible_v<dnn::Sigmoid>, "dnn::Sigmoid()");
}

TEST(ActivationHeaders, ScaledSigmoid) {
    static_assert(isActivationFunction<dnn::ScaledSigmoid>());
    static_assert(std::is_constructible_v<dnn::ScaledSigmoid, double, double>, "dnn::ScaledSigmoid(double min, double max)");
    static_assert(std::is_same_v<decltype(&dnn::ScaledSigmoid::getMin), double (dnn::ScaledSigmoid::*)() const>,
                  "double ScaledSigmoid::getMin() const");
    static_assert(std::is_same_v<decltype(&dnn::ScaledSigmoid::getMax), double (dnn::ScaledSigmoid::*)() const>,
                  "double ScaledSigmoid::getMax() const");
}

TEST(ActivationHeaders, ReLU) {
    static_assert(isActivationFunction<dnn::ReLU>());
    static_assert(std::is_default_constructible_v<dnn::ReLU>, "dnn::ReLU()");
}

TEST(ActivationHeaders, LeakyReLU) {
    static_assert(isActivationFunction<dnn::LeakyReLU>());
    static_assert(std::is_default_constructible_v<dnn::LeakyReLU>, "dnn::LeakyReLU()");
}
