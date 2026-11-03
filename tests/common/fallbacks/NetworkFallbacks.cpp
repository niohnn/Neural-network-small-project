#include "Fallback.h"

#include <Network.h>

namespace dnn {

DNN_FALLBACK Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>>,
                              LossFunction*) {
    fallback::notImplemented("dnn::Network::Network", 9);
}

// Destructors must not throw: without ~Network() the layers and the loss function are leaked (unit test 9 detects that).
DNN_FALLBACK Network::~Network() {}

DNN_FALLBACK Network& Network::addLayer(uint64_t, uint64_t, ActivationFunction*) {
    fallback::notImplemented("dnn::Network::addLayer", 10);
}

DNN_FALLBACK const std::vector<double>& Network::getLayerInput(uint64_t) const {
    fallback::notImplemented("dnn::Network::getLayerInput", 10);
}

DNN_FALLBACK void Network::fullForwardPropagation(const std::vector<double>&) {
    fallback::notImplemented("dnn::Network::fullForwardPropagation", 11);
}

DNN_FALLBACK void Network::fullBackwardPropagation(const std::vector<double>&, const std::vector<double>&) {
    fallback::notImplemented("dnn::Network::fullBackwardPropagation", 11);
}

DNN_FALLBACK void Network::trainBatch(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&, std::size_t, int,
                                      double) {
    fallback::notImplemented("dnn::Network::trainBatch", 12);
}

DNN_FALLBACK double Network::computeAccuracy(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&) {
    fallback::notImplemented("dnn::Network::computeAccuracy", 12);
}

}
