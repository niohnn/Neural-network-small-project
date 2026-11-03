#include "LossTestUtils.h"
#include "RefBridge.h"

#include <loss/BinaryCrossEntropy.h>

using namespace dnntest;

TEST(BinaryCrossEntropy, GetAccuracyValue) {
    dnn::BinaryCrossEntropy bce;
    const dnn::LossFunction& loss = bce;
    for (const LossCase& c : binaryCrossEntropyCases()) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(valueClose("getAccuracyValue()", loss.getAccuracyValue(c.predicted, c.actual), dnnref::bceAccuracy(c.predicted, c.actual),
                               kExact));
    }
}

TEST(BinaryCrossEntropy, GetGradient) {
    dnn::BinaryCrossEntropy bce;
    const dnn::LossFunction& loss = bce;
    for (const LossCase& c : binaryCrossEntropyCases()) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(vecClose("getGradient()", loss.getGradient(c.predicted, c.actual), dnnref::bceGradient(c.predicted, c.actual)));
    }
}
