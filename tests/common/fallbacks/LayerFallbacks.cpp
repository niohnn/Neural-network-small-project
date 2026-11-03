#include "Fallback.h"

#include <Layer.h>

namespace dnn {

DNN_FALLBACK void Layer::forwardPropagation(const std::vector<double>&) {
    fallback::notImplemented("dnn::Layer::forwardPropagation", 5);
}

DNN_FALLBACK void Layer::backwardPropagation(const std::vector<double>&, const std::vector<double>&) {
    fallback::notImplemented("dnn::Layer::backwardPropagation", 6);
}

DNN_FALLBACK void Layer::updateParameters(const std::vector<std::vector<double>>&, const std::vector<double>&, double) {
    fallback::notImplemented("dnn::Layer::updateParameters", 7);
}

DNN_FALLBACK void Layer::commitToWeights(double) {
    fallback::notImplemented("dnn::Layer::commitToWeights", 8);
}

}
