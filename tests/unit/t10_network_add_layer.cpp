#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

#include <Network.h>

using namespace dnntest;

namespace {

// Networks start from the provided Network() / setLayers(), so this test does not depend on the
// constructor. The state of a forward pass is set directly, so getLayerInput() is tested without
// fullForwardPropagation().
constexpr unsigned kFeatures = feature::NetworkAddLayer | feature::NetworkLayerInput;

struct Case {
    std::string name;
    NetworkSpec spec;
    std::vector<NetworkOp> ops;
};

void expectMatchesReference(const std::vector<Case>& cases) {
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        // init() uses the same seeds and generator; a tolerance (instead of exact equality) allows for
        // different compiler flags of the implementation under test and the prebuilt reference
        expectNetworkTrace(dnn::harness::runNetwork<kFeatures>(c.spec, c.ops), dnnref::runNetwork(c.spec, c.ops, kFeatures));
    }
}

}

TEST(NetworkAddLayer, AddLayer) {
    std::mt19937 rng(10);
    std::vector<Case> cases;
    {
        // weights come from Layer::init() with seeds 42, 43, ...
        NetworkSpec spec;
        spec.added = {{2, 4, {ActKind::TestTanh}}, {4, 3, {ActKind::TestPiecewiseLinear}}, {3, 1, {ActKind::TestTanh}}};
        spec.loss = LossKind::TestSquaredError;
        cases.push_back({"addLayer chain", spec, {}});
    }
    {
        NetworkSpec spec;
        spec.added = {{1, 1, {ActKind::TestTanh}}};
        cases.push_back({"single layer", spec, {}});
    }
    {
        NetworkSpec spec;
        spec.added = {{784, 16, {ActKind::TestTanh}}, {16, 10, {ActKind::TestPiecewiseLinear}}};
        cases.push_back({"large layers", spec, {}});
    }
    {
        // ids continue after the existing layers
        NetworkSpec spec;
        spec.pretrained = {randomLayer(rng, 3, 4, ActKind::TestTanh), randomLayer(rng, 4, 2, ActKind::TestTanh)};
        spec.added = {{2, 2, {ActKind::TestPiecewiseLinear}}, {2, 1, {ActKind::TestTanh}}};
        spec.loss = LossKind::TestSquaredError;
        cases.push_back({"existing layers followed by addLayer", spec, {}});
    }
    expectMatchesReference(cases);

    {
        SCOPED_TRACE("addLayer returns the network itself and takes ownership of the activation function");
        dnn::Network network;
        auto* activation = new dnn::TestTanh();
        dnn::Network& first = network.addLayer(2, 3, activation);
        EXPECT_EQ(&first, &network);
        dnn::Network& second = network.addLayer(3, 1, new dnn::TestTanh()).addLayer(1, 1, new dnn::TestTanh());
        EXPECT_EQ(&second, &network);
        ASSERT_EQ(network.getLayers().size(), 3u);
        EXPECT_EQ(network.getLayers()[0]->getActivationFunction(), activation);
    }
}

TEST(NetworkAddLayer, GetLayerInput) {
    std::mt19937 rng(100);
    std::vector<Case> cases;
    {
        NetworkSpec spec;
        spec.pretrained = {randomLayer(rng, 3, 4, ActKind::TestTanh), randomLayer(rng, 4, 2, ActKind::TestTanh),
                           randomLayer(rng, 2, 1, ActKind::TestPiecewiseLinear)};
        Case c{"three layers", spec, {}};
        // before any forward pass, then after two (simulated) forward passes
        c.ops = {NetworkOp::setLayerInputs(randomVec(rng, 3), {randomVec(rng, 4), randomVec(rng, 2), randomVec(rng, 1)}),
                 NetworkOp::setLayerInputs(randomVec(rng, 3), {randomVec(rng, 4), randomVec(rng, 2), randomVec(rng, 1)})};
        cases.push_back(c);
    }
    {
        NetworkSpec spec;
        spec.pretrained = {randomLayer(rng, 5, 2, ActKind::TestTanh)};
        Case c{"one layer", spec, {}};
        c.ops = {NetworkOp::setLayerInputs(randomVec(rng, 5), {randomVec(rng, 2)})};
        cases.push_back(c);
    }
    expectMatchesReference(cases);
}
