#pragma once

// Weak fallback definitions of the functions of MathUtils, Layer, Network and BinaryCrossEntropy that
// have to be implemented. The linker only uses a fallback if the implementation under test does not
// define the function, so a unit test links even while functions it does not test are missing; calling
// a missing function throws dnntest::NotImplemented. One file per class: the static library only links
// the files whose functions are missing (e.g. the BinaryCrossEntropy vtable needs getName() from
// BinaryCrossEntropy.cpp). Not used by test_compilation, which checks that every function is defined.

#include "HarnessTypes.h"

#include <string>

#define DNN_FALLBACK __attribute__((weak))

namespace dnn::fallback {

[[noreturn]] inline void notImplemented(const char* function, int test) {
    throw dnntest::NotImplemented(std::string(function) + " is not implemented (unit test " + std::to_string(test) + ")");
}

}
