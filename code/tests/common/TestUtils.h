#pragma once

// Comparison helpers and input generators for the unit tests.

#include "HarnessTypes.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dnntest {

struct Tolerance {
    double relative;
    double absolute;
};

// Allows for a different but equivalent order of floating-point operations.
constexpr Tolerance kDefaultTolerance{1e-10, 1e-12};
// trainBatch: rounding differences accumulate over many updates.
constexpr Tolerance kTrainingTolerance{1e-8, 1e-10};
constexpr Tolerance kExact{0.0, 0.0};

inline bool isClose(double actual, double expected, Tolerance tolerance) {
    if (std::isnan(actual) || std::isnan(expected)) {
        return std::isnan(actual) && std::isnan(expected);
    }
    if (actual == expected) {
        return true; // also covers equal infinities
    }
    return std::abs(actual - expected) <= tolerance.absolute + tolerance.relative * std::max(std::abs(actual), std::abs(expected));
}

inline std::string formatDouble(double value) {
    std::ostringstream stream;
    stream.precision(17);
    stream << value;
    return stream.str();
}

inline ::testing::AssertionResult valueClose(const char* what, double actual, double expected, Tolerance tolerance = kDefaultTolerance) {
    if (isClose(actual, expected, tolerance)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << what << ": got " << formatDouble(actual) << ", reference "
                                         << formatDouble(expected);
}

inline ::testing::AssertionResult vecClose(const char* what, const Vec& actual, const Vec& expected,
                                           Tolerance tolerance = kDefaultTolerance) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure() << what << ": size " << actual.size() << ", reference size " << expected.size();
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!isClose(actual[i], expected[i], tolerance)) {
            return ::testing::AssertionFailure() << what << "[" << i << "]: got " << formatDouble(actual[i])
                                                 << ", reference " << formatDouble(expected[i]);
        }
    }
    return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult matClose(const char* what, const Mat& actual, const Mat& expected,
                                           Tolerance tolerance = kDefaultTolerance) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure() << what << ": " << actual.size() << " rows, reference " << expected.size() << " rows";
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const std::string row = std::string(what) + "[" + std::to_string(i) + "]";
        ::testing::AssertionResult result = vecClose(row.c_str(), actual[i], expected[i], tolerance);
        if (!result) {
            return result;
        }
    }
    return ::testing::AssertionSuccess();
}

inline void expectLayerSnapshot(const LayerSnapshot& actual, const LayerSnapshot& expected, Tolerance tolerance) {
    EXPECT_EQ(toString(actual.error), toString(expected.error)) << "exception thrown";
    EXPECT_TRUE(matClose("weights", actual.weights, expected.weights, tolerance));
    EXPECT_TRUE(vecClose("bias", actual.bias, expected.bias, tolerance));
    EXPECT_TRUE(vecClose("preActivationOutput", actual.preActivationOutput, expected.preActivationOutput, tolerance));
    EXPECT_TRUE(vecClose("activationOutput", actual.activationOutput, expected.activationOutput, tolerance));
    EXPECT_TRUE(vecClose("dPreviousLayerOutput", actual.dPreviousLayerOutput, expected.dPreviousLayerOutput, tolerance));
    EXPECT_TRUE(matClose("dWeights", actual.dWeights, expected.dWeights, tolerance));
    EXPECT_TRUE(vecClose("dBias", actual.dBias, expected.dBias, tolerance));
    EXPECT_TRUE(matClose("accumulatedDWeights", actual.accumulatedDWeights, expected.accumulatedDWeights, tolerance));
    EXPECT_TRUE(vecClose("accumulatedDBias", actual.accumulatedDBias, expected.accumulatedDBias, tolerance));
    EXPECT_EQ(actual.numProcessedSamples, expected.numProcessedSamples) << "numProcessedSamples";
}

inline void expectLayerTrace(const LayerTrace& actual, const LayerTrace& expected, Tolerance tolerance = kDefaultTolerance) {
    ASSERT_EQ(actual.size(), expected.size()) << "trace length";
    for (std::size_t step = 0; step < actual.size(); ++step) {
        SCOPED_TRACE(step == 0 ? std::string("after construction") : "after op #" + std::to_string(step - 1));
        expectLayerSnapshot(actual[step], expected[step], tolerance);
    }
}

inline void expectNetworkTrace(const NetworkTrace& actual, const NetworkTrace& expected, Tolerance tolerance = kDefaultTolerance) {
    ASSERT_EQ(actual.size(), expected.size()) << "trace length";
    for (std::size_t step = 0; step < actual.size(); ++step) {
        SCOPED_TRACE(step == 0 ? std::string("after construction") : "after op #" + std::to_string(step - 1));
        const NetworkSnapshot& a = actual[step];
        const NetworkSnapshot& e = expected[step];
        EXPECT_EQ(toString(a.error), toString(e.error)) << "exception thrown";
        EXPECT_TRUE(valueClose("computeAccuracy", a.accuracy, e.accuracy, tolerance));
        EXPECT_EQ(a.layerIds, e.layerIds) << "layer ids";
        EXPECT_EQ(a.activationNames, e.activationNames) << "activation function per layer";
        EXPECT_TRUE(vecClose("output", a.output, e.output, tolerance));
        EXPECT_TRUE(matClose("layerInputs", a.layerInputs, e.layerInputs, tolerance));
        EXPECT_TRUE(vecClose("costHistory", a.costHistory, e.costHistory, tolerance));
        EXPECT_TRUE(vecClose("accuracyHistory", a.accuracyHistory, e.accuracyHistory, tolerance));
        ASSERT_EQ(a.layers.size(), e.layers.size()) << "number of layers";
        for (std::size_t i = 0; i < a.layers.size(); ++i) {
            SCOPED_TRACE("layer " + std::to_string(i));
            expectLayerSnapshot(a.layers[i], e.layers[i], tolerance);
        }
    }
}

// ---- input generators ----

inline Vec randomVec(std::mt19937& rng, std::size_t size, double lo = -1.0, double hi = 1.0) {
    std::uniform_real_distribution<double> distribution(lo, hi);
    Vec values(size);
    for (double& value : values) {
        value = distribution(rng);
    }
    return values;
}

inline Mat randomMat(std::mt19937& rng, std::size_t rows, std::size_t cols, double lo = -1.0, double hi = 1.0) {
    Mat values(rows);
    for (Vec& row : values) {
        row = randomVec(rng, cols, lo, hi);
    }
    return values;
}

inline std::vector<int> randomInts(std::mt19937& rng, std::size_t size, int lo, int hi) {
    std::uniform_int_distribution<int> distribution(lo, hi);
    std::vector<int> values(size);
    for (int& value : values) {
        value = distribution(rng);
    }
    return values;
}

inline PretrainedLayer randomLayer(std::mt19937& rng, std::size_t inputSize, std::size_t outputSize, ActKind activation) {
    return {randomMat(rng, outputSize, inputSize), randomVec(rng, outputSize, -0.5, 0.5), {activation}};
}

// Sample points for scalar activation functions: hand-picked boundary values plus random values.
inline Vec activationSamplePoints() {
    Vec points = {0.0, -0.0, 1e-300, -1e-300, 1e-8, -1e-8, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0,
                  10.0, -10.0, 40.0, -40.0, 100.0, -100.0, 709.0, -709.0, 710.0, -710.0, 1000.0, -1000.0};
    std::mt19937 rng(1234);
    const Vec random = randomVec(rng, 200, -20.0, 20.0);
    points.insert(points.end(), random.begin(), random.end());
    return points;
}

}
