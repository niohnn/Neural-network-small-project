#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

#include <Network.h>

using namespace dnntest;

namespace {

constexpr unsigned kFeatures = feature::NetworkPretrainedConstructor;

int deletedActivations = 0;
int deletedLosses = 0;

class CountingActivation : public dnn::TestTanh {
public:
    ~CountingActivation() override { ++deletedActivations; }
};

class CountingLoss : public dnn::TestSquaredError {
public:
    ~CountingLoss() override { ++deletedLosses; }
};

NetworkSpec pretrainedSpec(std::vector<PretrainedLayer> layers, LossKind loss = LossKind::TestSquaredError) {
    NetworkSpec spec;
    spec.construction = NetworkSpec::Construction::PretrainedConstructor;
    spec.pretrained = std::move(layers);
    spec.loss = loss;
    return spec;
}

}

TEST(NetworkConstruction, Constructor) {
    std::mt19937 rng(9);
    struct Case {
        std::string name;
        NetworkSpec spec;
    };
    const std::vector<Case> cases = {
        {"one layer", pretrainedSpec({randomLayer(rng, 3, 2, ActKind::TestTanh)})},
        {"three layers", pretrainedSpec({randomLayer(rng, 4, 5, ActKind::TestPiecewiseLinear), randomLayer(rng, 5, 3, ActKind::TestTanh),
                                         randomLayer(rng, 3, 1, ActKind::TestPiecewiseLinear)})},
        {"single input and output", pretrainedSpec({randomLayer(rng, 1, 1, ActKind::TestTanh)})},
        {"large layers", pretrainedSpec({randomLayer(rng, 784, 16, ActKind::TestTanh), randomLayer(rng, 16, 10, ActKind::TestPiecewiseLinear)})},
        {"no loss function", pretrainedSpec({randomLayer(rng, 2, 2, ActKind::TestTanh)}, LossKind::None)},
        {"no layers", pretrainedSpec({})},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectNetworkTrace(dnn::harness::runNetwork<kFeatures>(c.spec, {}), dnnref::runNetwork(c.spec, {}, kFeatures));
    }

    {
        SCOPED_TRACE("takes ownership of the given activation and loss functions");
        auto* first = new dnn::TestTanh();
        auto* second = new dnn::TestPiecewiseLinear();
        auto* loss = new dnn::TestSquaredError();
        std::vector<std::tuple<Mat, Vec, dnn::ActivationFunction*>> layerParameters;
        layerParameters.emplace_back(randomMat(rng, 3, 2), randomVec(rng, 3), first);
        layerParameters.emplace_back(randomMat(rng, 1, 3), randomVec(rng, 1), second);
        dnn::Network network(std::move(layerParameters), loss);
        ASSERT_EQ(network.getLayers().size(), 2u);
        EXPECT_EQ(network.getLayers()[0]->getActivationFunction(), first);
        EXPECT_EQ(network.getLayers()[1]->getActivationFunction(), second);
        EXPECT_EQ(network.getLossFunction(), loss);
    }
}

TEST(NetworkConstruction, Destructor) {
    std::mt19937 rng(90);
    {
        // independent of the constructor: layers and loss function set via setLayers() / setLossFunction()
        SCOPED_TRACE("network built by setLayers() and setLossFunction()");
        deletedActivations = 0;
        deletedLosses = 0;
        {
            dnn::Network network;
            network.setLayers({new dnn::Layer(0, 0, 2, 2, new CountingActivation()), new dnn::Layer(1, 0, 2, 1, new CountingActivation())});
            network.setLossFunction(new CountingLoss());
        }
        EXPECT_EQ(deletedActivations, 2) << "deleted activation functions (each layer deletes its own)";
        EXPECT_EQ(deletedLosses, 1) << "deleted loss functions";
    }
    {
        SCOPED_TRACE("network built by the constructor");
        deletedActivations = 0;
        deletedLosses = 0;
        {
            std::vector<std::tuple<Mat, Vec, dnn::ActivationFunction*>> layerParameters;
            layerParameters.emplace_back(randomMat(rng, 3, 2), randomVec(rng, 3), new CountingActivation());
            layerParameters.emplace_back(randomMat(rng, 1, 3), randomVec(rng, 1), new CountingActivation());
            layerParameters.emplace_back(randomMat(rng, 2, 1), randomVec(rng, 2), new CountingActivation());
            dnn::Network network(std::move(layerParameters), new CountingLoss());
            EXPECT_EQ(deletedActivations, 0) << "deleted while the network is alive";
            EXPECT_EQ(deletedLosses, 0) << "deleted while the network is alive";
        }
        EXPECT_EQ(deletedActivations, 3) << "deleted activation functions (each layer deletes its own)";
        EXPECT_EQ(deletedLosses, 1) << "deleted loss functions";
    }
    {
        SCOPED_TRACE("empty network");
        deletedActivations = 0;
        deletedLosses = 0;
        {
            dnn::Network network;
        }
        EXPECT_EQ(deletedActivations, 0);
        EXPECT_EQ(deletedLosses, 0);
    }
}
