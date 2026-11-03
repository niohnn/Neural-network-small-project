#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

constexpr unsigned kFeatures = feature::LayerForward;

LayerSpec makeSpec(std::mt19937& rng, uint64_t inputSize, uint64_t outputSize, ActKind activation) {
    LayerSpec spec;
    spec.id = 3;
    spec.inputSize = inputSize;
    spec.outputSize = outputSize;
    spec.activation = {activation};
    spec.weights = randomMat(rng, outputSize, inputSize);
    spec.bias = randomVec(rng, outputSize, -0.5, 0.5);
    return spec;
}

}

TEST(LayerForward, MatchesReference) {
    std::mt19937 rng(9);
    struct Case {
        std::string name;
        LayerSpec spec;
        std::vector<LayerOp> ops;
    };
    std::vector<Case> cases;

    for (const ActKind activation : {ActKind::TestTanh, ActKind::TestPiecewiseLinear}) {
        const std::string suffix = activation == ActKind::TestTanh ? " (tanh)" : " (piecewise linear)";
        for (const auto& [in, out] : std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {3, 2}, {2, 5}, {784, 16}, {16, 1}}) {
            const std::string shape = std::to_string(in) + "->" + std::to_string(out);
            // several consecutive forward passes: each must overwrite the previous outputs
            cases.push_back({"setParameters " + shape + suffix, makeSpec(rng, in, out, activation),
                             {LayerOp::forward(randomVec(rng, in)), LayerOp::forward(randomVec(rng, in, -3.0, 3.0)),
                              LayerOp::forward(Vec(in, 0.0))}});
        }
        LayerSpec initSpec;
        initSpec.id = 0;
        initSpec.seed = 42;
        initSpec.inputSize = 4;
        initSpec.outputSize = 3;
        initSpec.activation = {activation};
        initSpec.useInit = true;
        cases.push_back({"init() 4->3" + suffix, initSpec, {LayerOp::forward(randomVec(rng, 4)), LayerOp::forward(randomVec(rng, 4))}});
    }

    LayerSpec zeroSpec = makeSpec(rng, 3, 3, ActKind::TestPiecewiseLinear);
    zeroSpec.weights = Mat(3, Vec(3, 0.0));
    zeroSpec.bias = {0.0, -1.0, 1.0};
    cases.push_back({"zero weights, bias only", zeroSpec, {LayerOp::forward({1.0, 2.0, 3.0})}});

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectLayerTrace(dnn::harness::runLayer<kFeatures>(c.spec, c.ops), dnnref::runLayer(c.spec, c.ops));
    }
}
