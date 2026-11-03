#pragma once

// Runs a script of Layer/Network operations and records the observable state after each step.
// The same code is compiled twice: once against the implementation under test (namespace dnn)
// and once inside the reference library (dnn renamed to dnn_ref), so both sides are driven
// identically and their traces can be compared.

#include "HarnessTypes.h"
#include "TestDoubles.h"

#include <Layer.h>
#include <Network.h>

#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace dnn {
namespace harness {

// Thrown when a script uses an operation whose feature was not enabled; never mapped to an ErrorKind.
struct FeatureDisabled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline ActivationFunction* makeActivation(const dnntest::ActSpec& spec) {
    switch (spec.kind) {
        case dnntest::ActKind::None: return nullptr;
        case dnntest::ActKind::TestTanh: return new TestTanh();
        case dnntest::ActKind::TestPiecewiseLinear: return new TestPiecewiseLinear();
        default: throw std::runtime_error("harness: only test-double activations are supported");
    }
}

inline LossFunction* makeLoss(dnntest::LossKind kind) {
    switch (kind) {
        case dnntest::LossKind::None: return nullptr;
        case dnntest::LossKind::TestSquaredError: return new TestSquaredError();
    }
    return nullptr;
}

template <typename F>
dnntest::ErrorKind captureError(F&& f) {
    try {
        f();
    } catch (const FeatureDisabled&) {
        throw;
    } catch (const dnntest::NotImplemented&) {
        throw;
    } catch (const std::invalid_argument&) {
        return dnntest::ErrorKind::InvalidArgument;
    } catch (const std::out_of_range&) {
        return dnntest::ErrorKind::OutOfRange;
    } catch (const std::logic_error&) {
        return dnntest::ErrorKind::LogicError;
    } catch (...) {
        return dnntest::ErrorKind::OtherException;
    }
    return dnntest::ErrorKind::None;
}

// Access to private members (the explicit instantiation below may name them): updateParameters() is
// tested directly, and the state set up by other functions is injected so that a test does not depend
// on them.
template <typename Tag, typename Tag::type Member>
struct PrivateMember {
    friend typename Tag::type get(Tag) { return Member; }
};

#define DNN_HARNESS_PRIVATE_MEMBER(tagName, memberType, member)  \
    struct tagName {                                             \
        using type = memberType;                                 \
        friend type get(tagName);                                \
    };                                                           \
    template struct PrivateMember<tagName, &member>

DNN_HARNESS_PRIVATE_MEMBER(LayerUpdateParametersTag,
                           void (Layer::*)(const std::vector<std::vector<double>>&, const std::vector<double>&, double),
                           Layer::updateParameters);
DNN_HARNESS_PRIVATE_MEMBER(LayerPreActivationOutputTag, std::vector<double> Layer::*, Layer::preActivationOutput);
DNN_HARNESS_PRIVATE_MEMBER(LayerActivationOutputTag, std::vector<double> Layer::*, Layer::activationOutput);
DNN_HARNESS_PRIVATE_MEMBER(LayerAccumulatedDWeightsTag, std::vector<std::vector<double>> Layer::*, Layer::accumulatedDWeights);
DNN_HARNESS_PRIVATE_MEMBER(LayerAccumulatedDBiasTag, std::vector<double> Layer::*, Layer::accumulatedDBias);
DNN_HARNESS_PRIVATE_MEMBER(LayerNumProcessedSamplesTag, uint64_t Layer::*, Layer::numProcessedSamples);
DNN_HARNESS_PRIVATE_MEMBER(NetworkInitialInputTag, std::vector<double> Network::*, Network::initialInput);

#undef DNN_HARNESS_PRIVATE_MEMBER

// Sets Z_curr and A_curr = activation(Z_curr), as forwardPropagation() would.
inline void setPreActivation(Layer& layer, const std::vector<double>& preActivationOutput) {
    layer.*get(LayerPreActivationOutputTag()) = preActivationOutput;
    std::vector<double> activationOutput;
    if (ActivationFunction* activationFunction = layer.getActivationFunction()) {
        for (const double z : preActivationOutput) {
            activationOutput.push_back(activationFunction->activation(z));
        }
    }
    layer.*get(LayerActivationOutputTag()) = activationOutput;
}

template <unsigned Features>
void requireFeature(unsigned feature, const char* name) {
    if ((Features & feature) == 0) {
        throw FeatureDisabled(std::string("harness: feature not enabled for this test: ") + name);
    }
}

// Silences std::cout (trainBatch prints its progress) for the lifetime of the object.
class CoutSilencer {
public:
    CoutSilencer() : previous(std::cout.rdbuf(sink.rdbuf())) {}
    ~CoutSilencer() { std::cout.rdbuf(previous); }
    CoutSilencer(const CoutSilencer&) = delete;
    CoutSilencer& operator=(const CoutSilencer&) = delete;

private:
    std::ostringstream sink;
    std::streambuf* previous;
};

inline dnntest::LayerSnapshot snapshotLayer(const Layer& layer, dnntest::ErrorKind error) {
    dnntest::LayerSnapshot snapshot;
    snapshot.error = error;
    snapshot.weights = layer.getWeights();
    snapshot.bias = layer.getBias();
    snapshot.preActivationOutput = layer.getPreActivationOutput();
    snapshot.activationOutput = layer.getActivationOutput();
    snapshot.dPreviousLayerOutput = layer.getDPreviousLayerOutput();
    snapshot.dWeights = layer.getDWeights();
    snapshot.dBias = layer.getDBias();
    snapshot.accumulatedDWeights = layer.*get(LayerAccumulatedDWeightsTag());
    snapshot.accumulatedDBias = layer.*get(LayerAccumulatedDBiasTag());
    snapshot.numProcessedSamples = layer.*get(LayerNumProcessedSamplesTag());
    return snapshot;
}

template <unsigned Features>
dnntest::LayerTrace runLayer(const dnntest::LayerSpec& spec, const std::vector<dnntest::LayerOp>& ops) {
    dnntest::LayerTrace trace;
    Layer layer(spec.id, spec.seed, spec.inputSize, spec.outputSize, makeActivation(spec.activation));
    if (spec.useInit) {
        layer.init();
    } else {
        layer.setParameters(spec.weights, spec.bias);
    }
    trace.push_back(snapshotLayer(layer, dnntest::ErrorKind::None));

    for (const dnntest::LayerOp& op : ops) {
        const dnntest::ErrorKind error = captureError([&] {
            switch (op.type) {
                case dnntest::LayerOp::Type::Forward:
                    requireFeature<Features>(dnntest::feature::LayerForward, "LayerForward");
                    if constexpr ((Features & dnntest::feature::LayerForward) != 0) {
                        layer.forwardPropagation(op.input);
                    }
                    break;
                case dnntest::LayerOp::Type::Backward:
                    requireFeature<Features>(dnntest::feature::LayerBackward, "LayerBackward");
                    if constexpr ((Features & dnntest::feature::LayerBackward) != 0) {
                        layer.backwardPropagation(op.dActivation, op.input);
                    }
                    break;
                case dnntest::LayerOp::Type::UpdateParameters:
                    requireFeature<Features>(dnntest::feature::LayerUpdateParameters, "LayerUpdateParameters");
                    if constexpr ((Features & dnntest::feature::LayerUpdateParameters) != 0) {
                        (layer.*get(LayerUpdateParametersTag()))(op.dWeights, op.dBias, op.learningRate);
                    }
                    break;
                case dnntest::LayerOp::Type::Commit:
                    requireFeature<Features>(dnntest::feature::LayerCommit, "LayerCommit");
                    if constexpr ((Features & dnntest::feature::LayerCommit) != 0) {
                        layer.commitToWeights(op.learningRate);
                    }
                    break;
                case dnntest::LayerOp::Type::SetPreActivation:
                    setPreActivation(layer, op.input);
                    break;
                case dnntest::LayerOp::Type::SetAccumulated:
                    layer.*get(LayerAccumulatedDWeightsTag()) = op.dWeights;
                    layer.*get(LayerAccumulatedDBiasTag()) = op.dBias;
                    layer.*get(LayerNumProcessedSamplesTag()) = op.sampleCount;
                    break;
            }
        });
        trace.push_back(snapshotLayer(layer, error));
    }
    return trace;
}

template <unsigned Features>
dnntest::NetworkSnapshot snapshotNetwork(const Network& network, dnntest::ErrorKind error, double accuracy) {
    dnntest::NetworkSnapshot snapshot;
    snapshot.error = error;
    snapshot.accuracy = accuracy;
    for (const Layer* layer : network.getLayers()) {
        snapshot.layerIds.push_back(layer->getId());
        const ActivationFunction* activationFunction = layer->getActivationFunction();
        snapshot.activationNames.push_back(activationFunction ? activationFunction->getName() : "");
        snapshot.layers.push_back(snapshotLayer(*layer, dnntest::ErrorKind::None));
    }
    snapshot.output = network.getOutput();
    snapshot.costHistory = network.getCostHistory();
    snapshot.accuracyHistory = network.getAccuracyHistory();
    if constexpr ((Features & dnntest::feature::NetworkLayerInput) != 0) {
        for (uint64_t i = 0; i < network.getLayers().size(); ++i) {
            snapshot.layerInputs.push_back(network.getLayerInput(i));
        }
    }
    return snapshot;
}

template <unsigned Features>
std::unique_ptr<Network> constructNetwork(const dnntest::NetworkSpec& spec) {
    std::unique_ptr<Network> network;
    switch (spec.construction) {
        case dnntest::NetworkSpec::Construction::SetLayers: {
            network = std::make_unique<Network>();
            std::vector<Layer*> layers;
            for (const dnntest::PretrainedLayer& parameters : spec.pretrained) {
                Layer* layer = new Layer(layers.size(), /*seed=*/0, parameters.weights[0].size(), parameters.weights.size(),
                                         makeActivation(parameters.activation));
                layer->setParameters(parameters.weights, parameters.bias);
                layers.push_back(layer);
            }
            network->setLayers(std::move(layers));
            network->setLossFunction(makeLoss(spec.loss));
            break;
        }
        case dnntest::NetworkSpec::Construction::PretrainedConstructor:
            requireFeature<Features>(dnntest::feature::NetworkPretrainedConstructor, "NetworkPretrainedConstructor");
            if constexpr ((Features & dnntest::feature::NetworkPretrainedConstructor) != 0) {
                std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters;
                for (const dnntest::PretrainedLayer& parameters : spec.pretrained) {
                    layerParameters.emplace_back(parameters.weights, parameters.bias, makeActivation(parameters.activation));
                }
                network = std::make_unique<Network>(std::move(layerParameters), makeLoss(spec.loss));
            }
            break;
    }
    if (!spec.added.empty()) {
        requireFeature<Features>(dnntest::feature::NetworkAddLayer, "NetworkAddLayer");
        if constexpr ((Features & dnntest::feature::NetworkAddLayer) != 0) {
            for (const dnntest::AddedLayer& layer : spec.added) {
                network->addLayer(layer.inputDim, layer.outputDim, makeActivation(layer.activation));
            }
        }
    }
    return network;
}

template <unsigned Features>
dnntest::NetworkTrace runNetwork(const dnntest::NetworkSpec& spec, const std::vector<dnntest::NetworkOp>& ops) {
    dnntest::NetworkTrace trace;
    std::unique_ptr<Network> network;

    const dnntest::ErrorKind constructionError = captureError([&] { network = constructNetwork<Features>(spec); });
    if (constructionError != dnntest::ErrorKind::None || !network) {
        dnntest::NetworkSnapshot snapshot;
        snapshot.error = constructionError;
        trace.push_back(snapshot);
        return trace;
    }
    trace.push_back(snapshotNetwork<Features>(*network, dnntest::ErrorKind::None, std::numeric_limits<double>::quiet_NaN()));

    for (const dnntest::NetworkOp& op : ops) {
        double accuracy = std::numeric_limits<double>::quiet_NaN();
        const dnntest::ErrorKind error = captureError([&] {
            switch (op.type) {
                case dnntest::NetworkOp::Type::Forward:
                    requireFeature<Features>(dnntest::feature::NetworkForward, "NetworkForward");
                    if constexpr ((Features & dnntest::feature::NetworkForward) != 0) {
                        network->fullForwardPropagation(op.input);
                    }
                    break;
                case dnntest::NetworkOp::Type::Backward:
                    requireFeature<Features>(dnntest::feature::NetworkBackward, "NetworkBackward");
                    if constexpr ((Features & dnntest::feature::NetworkBackward) != 0) {
                        const std::vector<double> predicted = op.predicted.empty() ? network->getOutput() : op.predicted;
                        network->fullBackwardPropagation(predicted, op.actual);
                    }
                    break;
                case dnntest::NetworkOp::Type::TrainBatch:
                    requireFeature<Features>(dnntest::feature::NetworkTrainBatch, "NetworkTrainBatch");
                    if constexpr ((Features & dnntest::feature::NetworkTrainBatch) != 0) {
                        CoutSilencer silencer;
                        network->trainBatch(op.inputs, op.actualOutputs, op.batchSize, op.epochs, op.learningRate);
                    }
                    break;
                case dnntest::NetworkOp::Type::ComputeAccuracy:
                    requireFeature<Features>(dnntest::feature::NetworkComputeAccuracy, "NetworkComputeAccuracy");
                    if constexpr ((Features & dnntest::feature::NetworkComputeAccuracy) != 0) {
                        accuracy = network->computeAccuracy(op.inputs, op.actualOutputs);
                    }
                    break;
                case dnntest::NetworkOp::Type::SetLayerInputs:
                    (*network).*get(NetworkInitialInputTag()) = op.input;
                    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
                        network->getLayers()[i]->*get(LayerActivationOutputTag()) = op.inputs[i];
                    }
                    break;
            }
        });
        trace.push_back(snapshotNetwork<Features>(*network, error, accuracy));
    }
    return trace;
}

}
}
