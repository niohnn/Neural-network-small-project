// The reference library as a single translation unit (compiled with -Ddnn=dnn_ref). The reference
// implementation and the harness are wrapped in an anonymous namespace, so only the bridge functions
// in namespace dnnref (RefBridge.h) have external linkage: after stripping, the library contains no
// names of the classes and functions of the reference implementation.

// Every header included by the reference implementation, the harness and the bridge has to be
// included here first: a header that is first included inside the anonymous namespace below would
// declare its contents there.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// shared with the tests: must keep external linkage
#include "HarnessTypes.h"
#include "RefBridge.h"

namespace {
// lib/ is relative to the reference source directory (DNN_REF_SRC_DIR); the headers of the reference
// implementation are first included by these files
#include "lib/MathUtils.cpp"
#include "lib/activation/Sigmoid.cpp"
#include "lib/activation/ScaledSigmoid.cpp"
#include "lib/activation/ReLU.cpp"
#include "lib/activation/LeakyReLU.cpp"
#include "lib/loss/BinaryCrossEntropy.cpp"
#include "lib/Layer.cpp"
#include "lib/Network.cpp"
#include "Harness.h"
}

#include "RefBridge.cpp"
