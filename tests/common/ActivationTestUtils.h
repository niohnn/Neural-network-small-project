#pragma once

#include "RefBridge.h"
#include "TestUtils.h"

#include <activation/ActivationFunction.h>

namespace dnntest {

// Compares activation(), activationBackwards() and getName() of an implementation under test,
// called through the ActivationFunction base class, with the reference for the same spec.
inline void expectActivationMatchesReference(dnn::ActivationFunction& function, const ActSpec& spec) {
    EXPECT_EQ(function.getName(), dnnref::activationName(spec)) << "getName()";
    for (const double x : activationSamplePoints()) {
        SCOPED_TRACE("x = " + formatDouble(x));
        EXPECT_TRUE(valueClose("activation(x)", function.activation(x), dnnref::activation(spec, x)));
        EXPECT_TRUE(valueClose("activationBackwards(x)", function.activationBackwards(x), dnnref::activationBackwards(spec, x)));
    }
}

}
