#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

// updateParameters() is private; the harness calls it directly with the given gradients.
constexpr unsigned kFeatures = feature::LayerUpdateParameters;

LayerSpec makeSpec(std::mt19937& rng, uint64_t inputSize, uint64_t outputSize) {
    LayerSpec spec;
    spec.id = 4;
    spec.inputSize = inputSize;
    spec.outputSize = outputSize;
    spec.activation = {ActKind::TestTanh};
    spec.weights = randomMat(rng, outputSize, inputSize);
    spec.bias = randomVec(rng, outputSize, -0.5, 0.5);
    return spec;
}

LayerOp randomUpdate(std::mt19937& rng, const LayerSpec& spec, double learningRate) {
    return LayerOp::updateParameters(randomMat(rng, spec.outputSize, spec.inputSize, -2.0, 2.0), randomVec(rng, spec.outputSize, -2.0, 2.0),
                                     learningRate);
}

}

TEST(LayerUpdateParameters, MatchesReference) {
    std::mt19937 rng(7);
    struct Case {
        std::string name;
        LayerSpec spec;
        std::vector<LayerOp> ops;
    };
    std::vector<Case> cases;

    for (const double learningRate : {0.0, 0.01, 0.5, 1.0, -0.1}) {
        for (const auto& [in, out] : std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {3, 2}, {2, 5}, {64, 10}}) {
            Case c{"lr " + formatDouble(learningRate) + ", " + std::to_string(in) + "->" + std::to_string(out), makeSpec(rng, in, out), {}};
            c.ops = {randomUpdate(rng, c.spec, learningRate)};
            cases.push_back(c);
        }
    }
    {
        Case c{"zero gradients", makeSpec(rng, 3, 2), {}};
        c.ops = {LayerOp::updateParameters({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, {0.0, 0.0}, 0.5)};
        cases.push_back(c);
    }
    {
        // weights and bias are updated independently of each other
        Case c{"only weight / only bias gradients", makeSpec(rng, 2, 2), {}};
        c.ops = {LayerOp::updateParameters({{1.0, -1.0}, {0.5, 2.0}}, {0.0, 0.0}, 0.1),
                 LayerOp::updateParameters({{0.0, 0.0}, {0.0, 0.0}}, {1.0, -3.0}, 0.1)};
        cases.push_back(c);
    }
    {
        Case c{"consecutive updates", makeSpec(rng, 4, 3), {}};
        for (const double learningRate : {0.1, 0.2, 0.05}) {
            c.ops.push_back(randomUpdate(rng, c.spec, learningRate));
        }
        cases.push_back(c);
    }
    {
        // the given gradients are used, not the stored ones (dWeights/dBias) or the accumulated ones
        Case c{"ignores stored gradients", makeSpec(rng, 3, 2), {}};
        c.ops = {LayerOp::setAccumulated(randomMat(rng, 2, 3), randomVec(rng, 2), 2), randomUpdate(rng, c.spec, 0.3)};
        cases.push_back(c);
    }
    {
        Case c{"large layer", makeSpec(rng, 784, 16), {}};
        c.ops = {randomUpdate(rng, c.spec, 0.05)};
        cases.push_back(c);
    }
    {
        LayerSpec spec;
        spec.seed = 7;
        spec.inputSize = 6;
        spec.outputSize = 4;
        spec.activation = {ActKind::TestTanh};
        spec.useInit = true;
        Case c{"init() weights", spec, {}};
        c.ops = {randomUpdate(rng, c.spec, 0.1)};
        cases.push_back(c);
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectLayerTrace(dnn::harness::runLayer<kFeatures>(c.spec, c.ops), dnnref::runLayer(c.spec, c.ops));
    }
}
