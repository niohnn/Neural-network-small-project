#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

// The gradients accumulated by backwardPropagation() are set directly, so this test only depends on
// updateParameters() (called by commitToWeights()).
constexpr unsigned kFeatures = feature::LayerCommit;

LayerSpec makeSpec(std::mt19937& rng, uint64_t inputSize, uint64_t outputSize, ActKind activation) {
    LayerSpec spec;
    spec.id = 2;
    spec.inputSize = inputSize;
    spec.outputSize = outputSize;
    spec.activation = {activation};
    spec.weights = randomMat(rng, outputSize, inputSize);
    spec.bias = randomVec(rng, outputSize, -0.5, 0.5);
    return spec;
}

// the state after backwardPropagation() of `count` samples
void appendSamples(std::mt19937& rng, const LayerSpec& spec, int count, std::vector<LayerOp>& ops) {
    const double scale = count;
    ops.push_back(LayerOp::setAccumulated(randomMat(rng, spec.outputSize, spec.inputSize, -scale, scale),
                                          randomVec(rng, spec.outputSize, -scale, scale), count));
}

}

TEST(LayerCommit, MatchesReference) {
    std::mt19937 rng(11);
    struct Case {
        std::string name;
        LayerSpec spec;
        std::vector<LayerOp> ops;
    };
    std::vector<Case> cases;

    {
        Case c{"commit without samples is a no-op", makeSpec(rng, 3, 2, ActKind::TestTanh), {}};
        c.ops = {LayerOp::commit(0.5), LayerOp::commit(1.0)};
        cases.push_back(c);
    }
    for (const double learningRate : {0.0, 0.01, 0.5, 1.0}) {
        for (const int samples : {1, 3, 8}) {
            Case c{"lr " + formatDouble(learningRate) + ", " + std::to_string(samples) + " sample(s) per commit",
                   makeSpec(rng, 4, 3, samples == 3 ? ActKind::TestPiecewiseLinear : ActKind::TestTanh), {}};
            appendSamples(rng, c.spec, samples, c.ops);
            c.ops.push_back(LayerOp::commit(learningRate));
            cases.push_back(c);
        }
    }
    {
        // accumulators must be reset after a commit
        Case c{"consecutive batches", makeSpec(rng, 5, 2, ActKind::TestTanh), {}};
        appendSamples(rng, c.spec, 2, c.ops);
        c.ops.push_back(LayerOp::commit(0.1));
        appendSamples(rng, c.spec, 3, c.ops);
        c.ops.push_back(LayerOp::commit(0.1));
        appendSamples(rng, c.spec, 1, c.ops);
        c.ops.push_back(LayerOp::commit(0.2));
        cases.push_back(c);
    }
    {
        Case c{"second commit without new samples is a no-op", makeSpec(rng, 3, 3, ActKind::TestPiecewiseLinear), {}};
        appendSamples(rng, c.spec, 2, c.ops);
        c.ops.push_back(LayerOp::commit(0.3));
        c.ops.push_back(LayerOp::commit(0.3));
        cases.push_back(c);
    }
    {
        Case c{"large layer", makeSpec(rng, 784, 16, ActKind::TestTanh), {}};
        appendSamples(rng, c.spec, 4, c.ops);
        c.ops.push_back(LayerOp::commit(0.05));
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
        appendSamples(rng, c.spec, 3, c.ops);
        c.ops.push_back(LayerOp::commit(0.1));
        cases.push_back(c);
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectLayerTrace(dnn::harness::runLayer<kFeatures>(c.spec, c.ops), dnnref::runLayer(c.spec, c.ops));
    }
}
