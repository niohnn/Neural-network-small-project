#include "Harness.h"
#include "RefBridge.h"
#include "TestUtils.h"

using namespace dnntest;

namespace {

// Networks are built via the provided Network::setLayers(), so this test does not depend on the
// constructor or addLayer().
constexpr unsigned kFeatures = feature::NetworkTrainBatch | feature::NetworkComputeAccuracy;

struct Dataset {
    Mat inputs;
    Mat outputs;
};

Dataset makeDataset(std::mt19937& rng, std::size_t samples, std::size_t inputSize, std::size_t outputSize) {
    Dataset dataset;
    std::bernoulli_distribution label(0.5);
    for (std::size_t s = 0; s < samples; ++s) {
        dataset.inputs.push_back(randomVec(rng, inputSize, -1.0, 1.0));
        Vec output(outputSize);
        for (double& value : output) {
            value = label(rng) ? 1.0 : 0.0;
        }
        dataset.outputs.push_back(output);
    }
    return dataset;
}

NetworkSpec makeSpec(std::mt19937& rng, LossKind loss = LossKind::TestSquaredError) {
    NetworkSpec spec;
    spec.pretrained = {randomLayer(rng, 3, 4, ActKind::TestTanh), randomLayer(rng, 4, 2, ActKind::TestPiecewiseLinear)};
    spec.loss = loss;
    return spec;
}

}

TEST(NetworkTraining, MatchesReference) {
    std::mt19937 rng(14);
    const Dataset train = makeDataset(rng, 7000, 30, 20);
    const Dataset test = makeDataset(rng, 5000, 30, 20);

    struct Case {
        std::string name;
        NetworkSpec spec;
        std::vector<NetworkOp> ops;
    };
    std::vector<Case> cases;

    // batch sizes below, equal to and above the dataset size
    for (const std::size_t batchSize : {1u, 2u, 3u, 7u, 10u}) {
        for (const int epochs : {1, 3}) {
            Case c{"batchSize " + std::to_string(batchSize) + ", " + std::to_string(epochs) + " epoch(s)", makeSpec(rng), {}};
            c.ops = {NetworkOp::trainBatch(train.inputs, train.outputs, batchSize, epochs, 0.1),
                     NetworkOp::computeAccuracy(train.inputs, train.outputs), NetworkOp::computeAccuracy(test.inputs, test.outputs)};
            cases.push_back(c);
        }
    }
    {
        Case c{"zero epochs", makeSpec(rng), {}};
        c.ops = {NetworkOp::trainBatch(train.inputs, train.outputs, 2, 0, 0.1), NetworkOp::computeAccuracy(test.inputs, test.outputs)};
        cases.push_back(c);
    }
    {
        // each call clears and repopulates the histories
        Case c{"repeated training", makeSpec(rng), {}};
        c.ops = {NetworkOp::trainBatch(train.inputs, train.outputs, 3, 4, 0.05), NetworkOp::trainBatch(train.inputs, train.outputs, 2, 2, 0.2),
                 NetworkOp::computeAccuracy(train.inputs, train.outputs)};
        cases.push_back(c);
    }
    {
        Case c{"accuracy without training", makeSpec(rng), {}};
        c.ops = {NetworkOp::computeAccuracy(test.inputs, test.outputs), NetworkOp::computeAccuracy({test.inputs[0]}, {test.outputs[0]})};
        cases.push_back(c);
    }
    {
        std::mt19937 bigRng(140);
        const Dataset big = makeDataset(bigRng, 40, 3, 2);
        Case c{"larger dataset, 10 epochs", makeSpec(rng), {}};
        c.ops = {NetworkOp::trainBatch(big.inputs, big.outputs, 8, 10, 0.05), NetworkOp::computeAccuracy(big.inputs, big.outputs)};
        cases.push_back(c);
    }
    {
        // invalid arguments: std::invalid_argument, without changing the network
        Case c{"invalid arguments", makeSpec(rng), {}};
        c.ops = {NetworkOp::trainBatch(train.inputs, train.outputs, 2, 1, 0.1),
                 NetworkOp::trainBatch({}, {}, 2, 1, 0.1),
                 NetworkOp::trainBatch(train.inputs, test.outputs, 2, 1, 0.1),
                 NetworkOp::trainBatch(train.inputs, train.outputs, 0, 1, 0.1),
                 NetworkOp::computeAccuracy({}, {}),
                 NetworkOp::computeAccuracy(train.inputs, test.outputs)};
        cases.push_back(c);
    }
    {
        // no loss function: std::logic_error, checked before the arguments
        Case c{"no loss function", makeSpec(rng, LossKind::None), {}};
        c.ops = {NetworkOp::trainBatch(train.inputs, train.outputs, 2, 1, 0.1), NetworkOp::trainBatch({}, {}, 0, 1, 0.1),
                 NetworkOp::computeAccuracy(test.inputs, test.outputs), NetworkOp::computeAccuracy({}, {})};
        cases.push_back(c);
    }

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        expectNetworkTrace(dnn::harness::runNetwork<kFeatures>(c.spec, c.ops), dnnref::runNetwork(c.spec, c.ops, kFeatures), kTrainingTolerance);
    }
}
