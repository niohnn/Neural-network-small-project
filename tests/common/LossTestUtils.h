#pragma once

#include "TestUtils.h"

#include <random>
#include <string>
#include <vector>

namespace dnntest {

struct LossCase {
    std::string name;
    Vec predicted;
    Vec actual;
};

// Inputs for the BinaryCrossEntropy tests.
inline std::vector<LossCase> binaryCrossEntropyCases() {
    std::vector<LossCase> cases;

    // Single outputs: boundary predictions (clamping, exactly 0.5) against valid and out-of-range labels.
    for (const double predicted : {0.0, 1.0, 0.5, 1e-12, 1.0 - 1e-12, 1e-10, 0.3, 0.7, 0.5000001, 0.4999999, -0.2, 1.3}) {
        for (const double actual : {0.0, 1.0, 0.5, 2.0, -1.0}) {
            cases.push_back({"p = " + formatDouble(predicted) + ", y = " + formatDouble(actual), {predicted}, {actual}});
        }
    }
    cases.push_back({"all correct", {0.9, 0.1, 0.8}, {1.0, 0.0, 1.0}});
    cases.push_back({"all wrong", {0.1, 0.9, 0.2}, {1.0, 0.0, 1.0}});
    cases.push_back({"one-hot", {0.05, 0.9, 0.02, 0.5, 0.01}, {0.0, 1.0, 0.0, 0.0, 0.0}});

    std::mt19937 rng(8);
    std::bernoulli_distribution label(0.3);
    for (const std::size_t size : {2u, 10u, 100u}) {
        Vec actual(size);
        for (double& value : actual) {
            value = label(rng) ? 1.0 : 0.0;
        }
        cases.push_back({"random size " + std::to_string(size), randomVec(rng, size, 0.0, 1.0), actual});
    }
    return cases;
}

}
