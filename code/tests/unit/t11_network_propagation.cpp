#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

// Networks are built via the provided Network::setLayers(), so this test does not depend on the
// constructor or addLayer().
constexpr unsigned kFeatures = feature::NetworkForward | feature::NetworkBackward | feature::NetworkLayerInput;

NetworkSpec makeSpec(std::vector<PretrainedLayer> layers, LossKind loss = LossKind::TestSquaredError) {
    NetworkSpec spec;
    spec.pretrained = std::move(layers);
    spec.loss = loss;
    return spec;
}

}

TEST(NetworkPropagation, MatchesReference) {
    std::mt19937 rng(13);
    struct Case {
        std::string name;
        NetworkSpec spec;
        std::vector<NetworkOp> ops;
    };
    std::vector<Case> cases;

    {
        Case c{"one layer", makeSpec({randomLayer(rng, 3, 2, ActKind::TestTanh)}), {}};
        c.ops = {NetworkOp::forward(randomVec(rng, 3)), NetworkOp::backward({1.0, 0.0})};
        cases.push_back(c);
    }
    {
        Case c{"three layers, several samples",
               makeSpec({randomLayer(rng, 4, 5, ActKind::TestPiecewiseLinear), randomLayer(rng, 5, 3, ActKind::TestTanh),
                         randomLayer(rng, 3, 2, ActKind::TestPiecewiseLinear)}),
               {}};
        for (int i = 0; i < 3; ++i) {
            c.ops.push_back(NetworkOp::forward(randomVec(rng, 4, -2.0, 2.0)));
            c.ops.push_back(NetworkOp::backward(randomVec(rng, 2, 0.0, 1.0)));
        }
        cases.push_back(c);
    }
    {
        Case c{"forward only, repeated", makeSpec({randomLayer(rng, 2, 4, ActKind::TestTanh), randomLayer(rng, 4, 1, ActKind::TestTanh)}), {}};
        c.ops = {NetworkOp::forward(randomVec(rng, 2)), NetworkOp::forward(randomVec(rng, 2)), NetworkOp::forward({0.0, 0.0})};
        cases.push_back(c);
    }
    {
        Case c{"MNIST-sized", makeSpec({randomLayer(rng, 784, 16, ActKind::TestTanh), randomLayer(rng, 16, 10, ActKind::TestPiecewiseLinear)}), {}};
        c.ops = {NetworkOp::forward(randomVec(rng, 784, 0.0, 1.0)), NetworkOp::backward(Vec{0, 0, 1, 0, 0, 0, 0, 0, 0, 0})};
        cases.push_back(c);
    }
    {
        // the backward pass must be seeded with the given prediction, not with getOutput()
        Case c{"explicit predicted output", makeSpec({randomLayer(rng, 3, 3, ActKind::TestTanh), randomLayer(rng, 3, 2, ActKind::TestTanh)}), {}};
        c.ops = {NetworkOp::forward(randomVec(rng, 3)), NetworkOp::backward({1.0, 0.0}, {0.25, 0.75})};
        cases.push_back(c);
    }
    {
        Case c{"no layers", makeSpec({}), {}};
        c.ops = {NetworkOp::forward({0.5, -0.5}), NetworkOp::backward({1.0, 0.0})};
        cases.push_back(c);
    }
    {
        // fullBackwardPropagation must throw std::logic_error without a loss function
        Case c{"no loss function", makeSpec({randomLayer(rng, 2, 2, ActKind::TestTanh)}, LossKind::None), {}};
        c.ops = {NetworkOp::forward(randomVec(rng, 2)), NetworkOp::backward({1.0, 0.0})};
        cases.push_back(c);
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectNetworkTrace(dnn::harness::runNetwork<kFeatures>(c.spec, c.ops), dnnref::runNetwork(c.spec, c.ops, kFeatures));
    }
}
