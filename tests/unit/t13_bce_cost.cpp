#include "LossTestUtils.h"
#include "RefBridge.h"

#include <loss/BinaryCrossEntropy.h>

using namespace dnntest;

TEST(BinaryCrossEntropy, GetCostValue) {
    dnn::BinaryCrossEntropy bce;
    const dnn::LossFunction& loss = bce;
    for (const LossCase& c : binaryCrossEntropyCases()) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(valueClose("getCostValue()", loss.getCostValue(c.predicted, c.actual), dnnref::bceCost(c.predicted, c.actual)));
    }
}
