// Interface check: compiles only if every class and function exists in namespace dnn with exactly the
// predefined signature, and links only if every function is defined. Nothing is checked at run time.

#include <IOUtils.h>
#include <Layer.h>
#include <MathUtils.h>
#include <Network.h>
#include <activation/ActivationFunction.h>
#include <activation/LeakyReLU.h>
#include <activation/ReLU.h>
#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>
#include <loss/BinaryCrossEntropy.h>
#include <loss/LossFunction.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// ---- MathUtils ----
static_assert(std::is_same_v<decltype(&dnn::dotProduct), double (*)(const Vec&, const Vec&)>, "dnn::dotProduct");
static_assert(std::is_same_v<decltype(&dnn::sum), double (*)(const Vec&)>, "dnn::sum");
static_assert(std::is_same_v<decltype(&dnn::encodeAsOneHot), Vec (*)(double, std::size_t)>, "dnn::encodeAsOneHot");
// Overloaded templates: initialising a pointer of the expected type only compiles if that overload exists.
[[maybe_unused]] Vec (*const normalizeInt)(const std::vector<int>&) = &dnn::normalize<int>;
[[maybe_unused]] Vec (*const normalizeDouble)(const Vec&) = &dnn::normalize<double>;
[[maybe_unused]] Vec (*const normalizeIntMax)(const std::vector<int>&, int) = &dnn::normalize<int>;
[[maybe_unused]] Vec (*const normalizeDoubleMax)(const Vec&, int) = &dnn::normalize<double>;

// ---- Activation functions ----
template <typename T>
constexpr bool isActivationFunction() {
    static_assert(std::is_base_of_v<dnn::ActivationFunction, T>, "must derive from dnn::ActivationFunction");
    static_assert(!std::is_abstract_v<T>, "must override activation, activationBackwards and getName");
    static_assert(std::is_same_v<decltype(&T::activation), double (T::*)(double)>, "double activation(double)");
    static_assert(std::is_same_v<decltype(&T::activationBackwards), double (T::*)(double)>, "double activationBackwards(double)");
    static_assert(std::is_same_v<decltype(&T::getName), std::string (T::*)() const>, "std::string getName() const");
    return true;
}
static_assert(isActivationFunction<dnn::Sigmoid>());
static_assert(isActivationFunction<dnn::ReLU>());
static_assert(isActivationFunction<dnn::LeakyReLU>());
static_assert(isActivationFunction<dnn::ScaledSigmoid>());
static_assert(std::is_default_constructible_v<dnn::Sigmoid>, "dnn::Sigmoid()");
static_assert(std::is_default_constructible_v<dnn::ReLU>, "dnn::ReLU()");
static_assert(std::is_default_constructible_v<dnn::LeakyReLU>, "dnn::LeakyReLU()");
static_assert(std::is_constructible_v<dnn::ScaledSigmoid, double, double>, "dnn::ScaledSigmoid(double, double)");
static_assert(std::is_same_v<decltype(&dnn::ScaledSigmoid::getMin), double (dnn::ScaledSigmoid::*)() const>, "ScaledSigmoid::getMin");
static_assert(std::is_same_v<decltype(&dnn::ScaledSigmoid::getMax), double (dnn::ScaledSigmoid::*)() const>, "ScaledSigmoid::getMax");

// ---- Loss functions ----
using BCE = dnn::BinaryCrossEntropy;
static_assert(std::is_base_of_v<dnn::LossFunction, BCE>, "BinaryCrossEntropy must derive from dnn::LossFunction");
static_assert(!std::is_abstract_v<BCE>, "BinaryCrossEntropy must override all LossFunction methods");
static_assert(std::is_default_constructible_v<BCE>, "dnn::BinaryCrossEntropy()");
static_assert(std::is_same_v<decltype(&BCE::getCostValue), double (BCE::*)(const Vec&, const Vec&) const>, "BCE::getCostValue");
static_assert(std::is_same_v<decltype(&BCE::getAccuracyValue), double (BCE::*)(const Vec&, const Vec&) const>, "BCE::getAccuracyValue");
static_assert(std::is_same_v<decltype(&BCE::getGradient), Vec (BCE::*)(const Vec&, const Vec&) const>, "BCE::getGradient");
static_assert(std::is_same_v<decltype(&BCE::getName), std::string (BCE::*)() const>, "BCE::getName");

// ---- Layer ----
using dnn::Layer;
static_assert(std::is_constructible_v<Layer, uint64_t, uint64_t, uint64_t, uint64_t, dnn::ActivationFunction*>, "Layer(id, seed, in, out, act)");
static_assert(std::is_constructible_v<Layer, uint64_t, uint64_t, uint64_t, uint64_t, dnn::ActivationFunction*, std::string>,
              "Layer(id, seed, in, out, act, name)");
static_assert(!std::is_copy_constructible_v<Layer> && !std::is_copy_assignable_v<Layer>, "Layer must not be copyable");
static_assert(std::is_same_v<decltype(&Layer::init), void (Layer::*)()>, "Layer::init");
static_assert(std::is_same_v<decltype(&Layer::setParameters), void (Layer::*)(Mat, Vec)>, "Layer::setParameters");
static_assert(std::is_same_v<decltype(&Layer::forwardPropagation), void (Layer::*)(const Vec&)>, "Layer::forwardPropagation");
static_assert(std::is_same_v<decltype(&Layer::backwardPropagation), void (Layer::*)(const Vec&, const Vec&)>, "Layer::backwardPropagation");
static_assert(std::is_same_v<decltype(&Layer::commitToWeights), void (Layer::*)(double)>, "Layer::commitToWeights");
static_assert(std::is_same_v<decltype(&Layer::getId), uint64_t (Layer::*)() const>, "Layer::getId");
static_assert(std::is_same_v<decltype(&Layer::getName), const std::string& (Layer::*)() const>, "Layer::getName");
static_assert(std::is_same_v<decltype(&Layer::getActivationOutput), const Vec& (Layer::*)() const>, "Layer::getActivationOutput");
static_assert(std::is_same_v<decltype(&Layer::getPreActivationOutput), const Vec& (Layer::*)() const>, "Layer::getPreActivationOutput");
static_assert(std::is_same_v<decltype(&Layer::getActivationFunction), dnn::ActivationFunction* (Layer::*)() const>, "Layer::getActivationFunction");
static_assert(std::is_same_v<decltype(&Layer::getWeights), const Mat& (Layer::*)() const>, "Layer::getWeights");
static_assert(std::is_same_v<decltype(&Layer::getBias), const Vec& (Layer::*)() const>, "Layer::getBias");
static_assert(std::is_same_v<decltype(&Layer::getDPreviousLayerOutput), const Vec& (Layer::*)() const>, "Layer::getDPreviousLayerOutput");
static_assert(std::is_same_v<decltype(&Layer::getDWeights), const Mat& (Layer::*)() const>, "Layer::getDWeights");
static_assert(std::is_same_v<decltype(&Layer::getDBias), const Vec& (Layer::*)() const>, "Layer::getDBias");

// ---- Network ----
using dnn::Network;
using LayerParameters = std::vector<std::tuple<Mat, Vec, dnn::ActivationFunction*>>;
static_assert(std::is_default_constructible_v<Network>, "Network()");
static_assert(std::is_constructible_v<Network, LayerParameters, dnn::LossFunction*>, "Network(layerParameters, lossFunction)");
static_assert(!std::is_copy_constructible_v<Network> && !std::is_copy_assignable_v<Network>, "Network must not be copyable");
static_assert(std::is_same_v<decltype(&Network::addLayer), Network& (Network::*)(uint64_t, uint64_t, dnn::ActivationFunction*)>, "Network::addLayer");
static_assert(std::is_same_v<decltype(&Network::fullForwardPropagation), void (Network::*)(const Vec&)>, "Network::fullForwardPropagation");
static_assert(std::is_same_v<decltype(&Network::fullBackwardPropagation), void (Network::*)(const Vec&, const Vec&)>,
              "Network::fullBackwardPropagation");
static_assert(std::is_same_v<decltype(&Network::trainBatch), void (Network::*)(const Mat&, const Mat&, std::size_t, int, double)>,
              "Network::trainBatch");
static_assert(std::is_same_v<decltype(&Network::computeAccuracy), double (Network::*)(const Mat&, const Mat&)>, "Network::computeAccuracy");
static_assert(std::is_same_v<decltype(&Network::setLossFunction), void (Network::*)(dnn::LossFunction*)>, "Network::setLossFunction");
static_assert(std::is_same_v<decltype(&Network::getLossFunction), const dnn::LossFunction* (Network::*)() const>, "Network::getLossFunction");
static_assert(std::is_same_v<decltype(&Network::getOutput), const Vec& (Network::*)() const>, "Network::getOutput");
static_assert(std::is_same_v<decltype(&Network::getLayerInput), const Vec& (Network::*)(uint64_t) const>, "Network::getLayerInput");
static_assert(std::is_same_v<decltype(&Network::getCostHistory), const Vec& (Network::*)() const>, "Network::getCostHistory");
static_assert(std::is_same_v<decltype(&Network::getAccuracyHistory), const Vec& (Network::*)() const>, "Network::getAccuracyHistory");
static_assert(std::is_same_v<decltype(&Network::getLayers), const std::vector<Layer*>& (Network::*)() const>, "Network::getLayers");
static_assert(std::is_same_v<decltype(&Network::setLayers), void (Network::*)(std::vector<Layer*>)>, "Network::setLayers");

// ---- IOUtils ----
static_assert(std::is_same_v<decltype(&dnn::saveModel), void (*)(const Network&, const std::string&)>, "dnn::saveModel");
static_assert(std::is_same_v<decltype(&dnn::loadModel), void (*)(Network&, const std::string&)>, "dnn::loadModel");

}

// Never called: it uses every function, so linking fails if one of them is declared but not defined.
void useEverything() {
    Vec v;
    Mat m;
    dnn::dotProduct(v, v);
    dnn::sum(v);
    dnn::encodeAsOneHot(0.0, 1);

    dnn::Sigmoid sigmoid;
    dnn::ReLU relu;
    dnn::LeakyReLU leakyRelu;
    dnn::ScaledSigmoid scaledSigmoid(0.0, 1.0);
    for (dnn::ActivationFunction* f : std::vector<dnn::ActivationFunction*>{&sigmoid, &relu, &leakyRelu, &scaledSigmoid}) {
        f->activation(0.0);
        f->activationBackwards(0.0);
        f->getName();
    }
    sigmoid.activation(0.0), sigmoid.activationBackwards(0.0), sigmoid.getName();
    relu.activation(0.0), relu.activationBackwards(0.0), relu.getName();
    leakyRelu.activation(0.0), leakyRelu.activationBackwards(0.0), leakyRelu.getName();
    scaledSigmoid.activation(0.0), scaledSigmoid.activationBackwards(0.0), scaledSigmoid.getName();
    scaledSigmoid.getMin(), scaledSigmoid.getMax();

    dnn::BinaryCrossEntropy bce;
    bce.getCostValue(v, v), bce.getAccuracyValue(v, v), bce.getGradient(v, v), bce.getName();

    dnn::Layer layer(0, 0, 1, 1, nullptr);
    layer.init();
    layer.setParameters(m, v);
    layer.forwardPropagation(v);
    layer.backwardPropagation(v, v);
    layer.commitToWeights(0.0);
    layer.getId(), layer.getName(), layer.getActivationOutput(), layer.getPreActivationOutput(), layer.getActivationFunction();
    layer.getWeights(), layer.getBias(), layer.getDPreviousLayerOutput(), layer.getDWeights(), layer.getDBias();

    dnn::Network network(LayerParameters{}, nullptr);
    dnn::Network other;
    network.addLayer(1, 1, nullptr);
    network.fullForwardPropagation(v);
    network.fullBackwardPropagation(v, v);
    network.trainBatch(m, m, 1, 1, 0.0);
    network.computeAccuracy(m, m);
    network.setLossFunction(nullptr);
    network.getLossFunction(), network.getOutput(), network.getLayerInput(0);
    network.getCostHistory(), network.getAccuracyHistory(), network.getLayers();
    network.setLayers({});

    dnn::saveModel(network, "");
    dnn::loadModel(network, "");
}

int main() {
    std::cout << "Interface check passed: all classes and functions exist with the predefined signatures.\n";
    return 0;
}
