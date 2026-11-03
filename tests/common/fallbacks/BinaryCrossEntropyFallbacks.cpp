#include "Fallback.h"

#include <loss/BinaryCrossEntropy.h>

namespace dnn {

// getCostValue() is the key function of BinaryCrossEntropy: with this fallback, the vtable is also emitted
// (as a weak symbol) if the implementation under test does not define getCostValue() yet.
DNN_FALLBACK double BinaryCrossEntropy::getCostValue(const std::vector<double>&, const std::vector<double>&) const {
    fallback::notImplemented("dnn::BinaryCrossEntropy::getCostValue", 13);
}

DNN_FALLBACK double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>&, const std::vector<double>&) const {
    fallback::notImplemented("dnn::BinaryCrossEntropy::getAccuracyValue", 14);
}

DNN_FALLBACK std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>&, const std::vector<double>&) const {
    fallback::notImplemented("dnn::BinaryCrossEntropy::getGradient", 14);
}

}
