#pragma once

// Plain data types shared between the tests and the reference bridge. They must not depend on
// anything in namespace dnn, since the reference library is compiled with dnn renamed to dnn_ref.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dnntest {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

enum class ActKind { None, Sigmoid, ScaledSigmoid, ReLU, LeakyReLU, TestTanh, TestPiecewiseLinear };

struct ActSpec {
    ActKind kind = ActKind::None;
    double min = 0.0; // only used by ScaledSigmoid
    double max = 1.0; // only used by ScaledSigmoid
};

enum class LossKind { None, TestSquaredError };

enum class ErrorKind { None, LogicError, InvalidArgument, OutOfRange, OtherException };

// Thrown by the fallback definitions (Fallbacks.cpp) of functions the implementation under test does not
// define yet. Not a std::logic_error, so it never counts as an expected exception.
struct NotImplemented : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline std::string toString(ErrorKind error) {
    switch (error) {
        case ErrorKind::None: return "none";
        case ErrorKind::LogicError: return "std::logic_error";
        case ErrorKind::InvalidArgument: return "std::invalid_argument";
        case ErrorKind::OutOfRange: return "std::out_of_range";
        case ErrorKind::OtherException: return "other exception";
    }
    return "?";
}

// Compile-time feature masks for the harness: an operation is only instantiated (and its function
// only linked) if its feature is enabled, so a test does not fail to link because a function it
// does not exercise is still missing.
namespace feature {
enum : unsigned {
    LayerForward = 1u << 0,
    LayerBackward = 1u << 1,
    LayerCommit = 1u << 2,
    NetworkPretrainedConstructor = 1u << 3,
    NetworkAddLayer = 1u << 4,
    NetworkForward = 1u << 5,
    NetworkBackward = 1u << 6,
    NetworkLayerInput = 1u << 7,
    NetworkTrainBatch = 1u << 8,
    NetworkComputeAccuracy = 1u << 9,
    LayerUpdateParameters = 1u << 10,
    All = ~0u,
};
}

// ---- Layer scripts ----

struct LayerSpec {
    uint64_t id = 0;
    uint64_t seed = 0;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    ActSpec activation;
    bool useInit = false; // true: Layer::init(), false: Layer::setParameters(weights, bias)
    Mat weights;
    Vec bias;
};

struct LayerOp {
    enum class Type {
        Forward,
        Backward,
        UpdateParameters,
        Commit,
        SetPreActivation, // sets the state of a forward pass (Z_curr, A_curr) without calling forwardPropagation()
        SetAccumulated,   // sets the gradients accumulated by backwardPropagation() without calling it
    };
    Type type = Type::Forward;
    Vec input;       // Forward, Backward: previousLayerOutput; SetPreActivation: Z_curr
    Vec dActivation; // Backward only
    Mat dWeights;    // UpdateParameters, SetAccumulated
    Vec dBias;       // UpdateParameters, SetAccumulated
    uint64_t sampleCount = 0;  // SetAccumulated: number of processed samples
    double learningRate = 0.0; // UpdateParameters, Commit

    static LayerOp forward(Vec input) {
        LayerOp op;
        op.type = Type::Forward;
        op.input = std::move(input);
        return op;
    }
    static LayerOp backward(Vec dActivation, Vec input) {
        LayerOp op;
        op.type = Type::Backward;
        op.input = std::move(input);
        op.dActivation = std::move(dActivation);
        return op;
    }
    static LayerOp updateParameters(Mat dWeights, Vec dBias, double learningRate) {
        LayerOp op;
        op.type = Type::UpdateParameters;
        op.dWeights = std::move(dWeights);
        op.dBias = std::move(dBias);
        op.learningRate = learningRate;
        return op;
    }
    static LayerOp commit(double learningRate) {
        LayerOp op;
        op.type = Type::Commit;
        op.learningRate = learningRate;
        return op;
    }
    static LayerOp setPreActivation(Vec preActivationOutput) {
        LayerOp op;
        op.type = Type::SetPreActivation;
        op.input = std::move(preActivationOutput);
        return op;
    }
    static LayerOp setAccumulated(Mat dWeights, Vec dBias, uint64_t sampleCount) {
        LayerOp op;
        op.type = Type::SetAccumulated;
        op.dWeights = std::move(dWeights);
        op.dBias = std::move(dBias);
        op.sampleCount = sampleCount;
        return op;
    }
};

struct LayerSnapshot {
    ErrorKind error = ErrorKind::None;
    Mat weights;
    Vec bias;
    Vec preActivationOutput;
    Vec activationOutput;
    Vec dPreviousLayerOutput;
    Mat dWeights;
    Vec dBias;
    // private state: gradients accumulated by backwardPropagation() for the next commitToWeights()
    Mat accumulatedDWeights;
    Vec accumulatedDBias;
    uint64_t numProcessedSamples = 0;
};

// One snapshot after construction, then one after every op.
using LayerTrace = std::vector<LayerSnapshot>;

// ---- Network scripts ----

struct PretrainedLayer {
    Mat weights;
    Vec bias;
    ActSpec activation;
};

struct AddedLayer {
    uint64_t inputDim = 0;
    uint64_t outputDim = 0;
    ActSpec activation;
};

struct NetworkSpec {
    enum class Construction {
        SetLayers,             // Network() + setLayers(Layers built from `pretrained`, seed 0) + setLossFunction(loss)
        PretrainedConstructor, // Network(pretrained, loss)
    };
    Construction construction = Construction::SetLayers;
    std::vector<PretrainedLayer> pretrained;
    std::vector<AddedLayer> added; // appended via addLayer() after construction
    LossKind loss = LossKind::None;
};

struct NetworkOp {
    enum class Type {
        Forward,
        Backward,
        TrainBatch,
        ComputeAccuracy,
        SetLayerInputs, // sets the state of a forward pass (network input, A_curr of each layer) without calling
                        // fullForwardPropagation()
    };
    Type type = Type::Forward;
    Vec input;              // Forward, SetLayerInputs: network input
    Vec predicted;          // Backward; empty means "use getOutput()"
    Vec actual;             // Backward
    Mat inputs;             // TrainBatch, ComputeAccuracy; SetLayerInputs: activation output of each layer
    Mat actualOutputs;      // TrainBatch, ComputeAccuracy
    std::size_t batchSize = 1;
    int epochs = 1;
    double learningRate = 0.0;

    static NetworkOp forward(Vec input) {
        NetworkOp op;
        op.type = Type::Forward;
        op.input = std::move(input);
        return op;
    }
    static NetworkOp backward(Vec actual, Vec predicted = {}) {
        NetworkOp op;
        op.type = Type::Backward;
        op.actual = std::move(actual);
        op.predicted = std::move(predicted);
        return op;
    }
    static NetworkOp setLayerInputs(Vec networkInput, Mat layerOutputs) {
        NetworkOp op;
        op.type = Type::SetLayerInputs;
        op.input = std::move(networkInput);
        op.inputs = std::move(layerOutputs);
        return op;
    }
    static NetworkOp trainBatch(Mat inputs, Mat actualOutputs, std::size_t batchSize, int epochs, double learningRate) {
        NetworkOp op;
        op.type = Type::TrainBatch;
        op.inputs = std::move(inputs);
        op.actualOutputs = std::move(actualOutputs);
        op.batchSize = batchSize;
        op.epochs = epochs;
        op.learningRate = learningRate;
        return op;
    }
    static NetworkOp computeAccuracy(Mat inputs, Mat actualOutputs) {
        NetworkOp op;
        op.type = Type::ComputeAccuracy;
        op.inputs = std::move(inputs);
        op.actualOutputs = std::move(actualOutputs);
        return op;
    }
};

struct NetworkSnapshot {
    ErrorKind error = ErrorKind::None;
    double accuracy = std::numeric_limits<double>::quiet_NaN(); // result of ComputeAccuracy, NaN otherwise
    std::vector<uint64_t> layerIds;
    std::vector<std::string> activationNames;
    std::vector<LayerSnapshot> layers;
    Vec output;
    Mat layerInputs; // getLayerInput(i) for every layer (only recorded with feature::NetworkLayerInput)
    Vec costHistory;
    Vec accuracyHistory;
};

// One snapshot after construction, then one after every op.
using NetworkTrace = std::vector<NetworkSnapshot>;

}
