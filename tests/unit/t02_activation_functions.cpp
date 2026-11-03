#include "ActivationTestUtils.h"

#include <activation/LeakyReLU.h>
#include <activation/ReLU.h>
#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>

#include <utility>

using namespace dnntest;

TEST(ActivationFunctions, Sigmoid) {
    dnn::Sigmoid sigmoid;
    expectActivationMatchesReference(sigmoid, {ActKind::Sigmoid});
}

TEST(ActivationFunctions, ScaledSigmoid) {
    const std::vector<std::pair<double, double>> ranges = {
        {0.0, 1.0}, {-1.0, 1.0}, {0.0, 255.0}, {2.0, 5.0}, {-10.0, -2.0}, {3.0, 3.0}, {1.0, -1.0},
    };
    for (const auto& [min, max] : ranges) {
        SCOPED_TRACE("min = " + formatDouble(min) + ", max = " + formatDouble(max));
        dnn::ScaledSigmoid scaledSigmoid(min, max);
        EXPECT_TRUE(valueClose("getMin()", scaledSigmoid.getMin(), dnnref::scaledSigmoidMin(min, max), kExact));
        EXPECT_TRUE(valueClose("getMax()", scaledSigmoid.getMax(), dnnref::scaledSigmoidMax(min, max), kExact));
        expectActivationMatchesReference(scaledSigmoid, {ActKind::ScaledSigmoid, min, max});
    }
}

TEST(ActivationFunctions, ReLU) {
    dnn::ReLU relu;
    expectActivationMatchesReference(relu, {ActKind::ReLU});
}

TEST(ActivationFunctions, LeakyReLU) {
    dnn::LeakyReLU leakyRelu;
    expectActivationMatchesReference(leakyRelu, {ActKind::LeakyReLU});
}
