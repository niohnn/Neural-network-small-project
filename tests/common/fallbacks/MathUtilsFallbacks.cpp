#include "Fallback.h"

#include <MathUtils.h>

namespace dnn {

DNN_FALLBACK double dotProduct(const std::vector<double>&, const std::vector<double>&) {
    fallback::notImplemented("dnn::dotProduct", 3);
}

DNN_FALLBACK double sum(const std::vector<double>&) {
    fallback::notImplemented("dnn::sum", 3);
}

}
