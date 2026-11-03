#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

// The state of the forward pass (Z_curr) is set directly, so this test does not depend on forwardPropagation().
constexpr unsigned kFeatures = feature::LayerBackward;

LayerSpec makeSpec(std::mt19937& rng, uint64_t inputSize, uint64_t outputSize, ActKind activation) {
    LayerSpec spec;
    spec.id = 1;
    spec.inputSize = inputSize;
    spec.outputSize = outputSize;
    spec.activation = {activation};
    spec.weights = randomMat(rng, outputSize, inputSize);
    spec.bias = randomVec(rng, outputSize, -0.5, 0.5);
    return spec;
}

// forward pass state + backward for one random sample
void appendSample(std::mt19937& rng, const LayerSpec& spec, std::vector<LayerOp>& ops) {
    const Vec input = randomVec(rng, spec.inputSize, -2.0, 2.0);
    ops.push_back(LayerOp::setPreActivation(randomVec(rng, spec.outputSize, -2.0, 2.0)));
    ops.push_back(LayerOp::backward(randomVec(rng, spec.outputSize), input));
}

}

TEST(LayerBackward, MatchesReference) {
    std::mt19937 rng(10);
    struct Case {
        std::string name;
        LayerSpec spec;
        std::vector<LayerOp> ops;
    };
    std::vector<Case> cases;

    for (const ActKind activation : {ActKind::TestTanh, ActKind::TestPiecewiseLinear}) {
        const std::string suffix = activation == ActKind::TestTanh ? " (tanh)" : " (piecewise linear)";
        for (const auto& [in, out] : std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {3, 2}, {2, 5}, {64, 10}}) {
            const std::string shape = std::to_string(in) + "->" + std::to_string(out);
            Case single{"one sample " + shape + suffix, makeSpec(rng, in, out, activation), {}};
            appendSample(rng, single.spec, single.ops);
            cases.push_back(single);

            // gradients of each backward pass must only depend on the current sample
            Case several{"several samples " + shape + suffix, makeSpec(rng, in, out, activation), {}};
            for (int i = 0; i < 4; ++i) {
                appendSample(rng, several.spec, several.ops);
            }
            cases.push_back(several);
        }
    }

    // backward twice for the same forward pass with different upstream gradients
    Case repeated{"repeated backward", makeSpec(rng, 3, 2, ActKind::TestTanh), {}};
    const Vec input = {0.5, -1.0, 2.0};
    repeated.ops = {LayerOp::setPreActivation({0.3, -1.2}), LayerOp::backward({1.0, 0.0}, input), LayerOp::backward({0.0, -2.0}, input)};
    cases.push_back(repeated);

    // zero upstream gradient
    Case zeroGradient{"zero upstream gradient", makeSpec(rng, 3, 2, ActKind::TestPiecewiseLinear), {}};
    zeroGradient.ops = {LayerOp::setPreActivation({0.7, -0.4}), LayerOp::backward({0.0, 0.0}, input)};
    cases.push_back(zeroGradient);

    // the derivative is evaluated at Z_curr: both branches of the piecewise linear test activation
    Case branches{"Z_curr around 0", makeSpec(rng, 2, 4, ActKind::TestPiecewiseLinear), {}};
    branches.ops = {LayerOp::setPreActivation({-1.0, -1e-9, 1e-9, 1.0}), LayerOp::backward({1.0, 1.0, 1.0, 1.0}, {0.5, -0.5})};
    cases.push_back(branches);

    // without activation function, backwardPropagation must throw std::logic_error
    Case noActivation{"no activation function", makeSpec(rng, 3, 2, ActKind::None), {}};
    noActivation.ops = {LayerOp::backward({1.0, 1.0}, input)};
    cases.push_back(noActivation);

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectLayerTrace(dnn::harness::runLayer<kFeatures>(c.spec, c.ops), dnnref::runLayer(c.spec, c.ops));
    }
}
