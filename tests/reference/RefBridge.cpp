// Compiled with -Ddnn=dnn_ref against the full reference headers: every dnn:: below refers to the
// reference implementation in namespace dnn_ref.

#include "RefBridge.h"

#include "Harness.h"

#include <MathUtils.h>
#include <activation/LeakyReLU.h>
#include <activation/ReLU.h>
#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>
#include <loss/BinaryCrossEntropy.h>

#include <memory>
#include <stdexcept>

namespace dnnref {

namespace {

std::unique_ptr<dnn::ActivationFunction> makeActivation(const dnntest::ActSpec& spec) {
    switch (spec.kind) {
        case dnntest::ActKind::Sigmoid: return std::make_unique<dnn::Sigmoid>();
        case dnntest::ActKind::ScaledSigmoid: return std::make_unique<dnn::ScaledSigmoid>(spec.min, spec.max);
        case dnntest::ActKind::ReLU: return std::make_unique<dnn::ReLU>();
        case dnntest::ActKind::LeakyReLU: return std::make_unique<dnn::LeakyReLU>();
        case dnntest::ActKind::TestTanh: return std::make_unique<dnn::TestTanh>();
        case dnntest::ActKind::TestPiecewiseLinear: return std::make_unique<dnn::TestPiecewiseLinear>();
        case dnntest::ActKind::None: break;
    }
    throw std::invalid_argument("dnnref: no activation function for ActKind::None");
}

}

double dotProduct(const Vec& a, const Vec& b) { return dnn::dotProduct(a, b); }
double sum(const Vec& values) { return dnn::sum(values); }
Vec normalize(const std::vector<int>& values) { return dnn::normalize(values); }
Vec normalize(const std::vector<double>& values) { return dnn::normalize(values); }
Vec normalize(const std::vector<int>& values, int maxValue) { return dnn::normalize(values, maxValue); }
Vec normalize(const std::vector<double>& values, int maxValue) { return dnn::normalize(values, maxValue); }

double activation(const dnntest::ActSpec& spec, double x) { return makeActivation(spec)->activation(x); }
double activationBackwards(const dnntest::ActSpec& spec, double x) { return makeActivation(spec)->activationBackwards(x); }
std::string activationName(const dnntest::ActSpec& spec) { return makeActivation(spec)->getName(); }
double scaledSigmoidMin(double min, double max) { return dnn::ScaledSigmoid(min, max).getMin(); }
double scaledSigmoidMax(double min, double max) { return dnn::ScaledSigmoid(min, max).getMax(); }

double bceCost(const Vec& predictedOutput, const Vec& actualOutput) {
    return dnn::BinaryCrossEntropy().getCostValue(predictedOutput, actualOutput);
}
double bceAccuracy(const Vec& predictedOutput, const Vec& actualOutput) {
    return dnn::BinaryCrossEntropy().getAccuracyValue(predictedOutput, actualOutput);
}
Vec bceGradient(const Vec& predictedOutput, const Vec& actualOutput) {
    return dnn::BinaryCrossEntropy().getGradient(predictedOutput, actualOutput);
}

dnntest::LayerTrace runLayer(const dnntest::LayerSpec& spec, const std::vector<dnntest::LayerOp>& ops) {
    return dnn::harness::runLayer<dnntest::feature::All>(spec, ops);
}

dnntest::NetworkTrace runNetwork(const dnntest::NetworkSpec& spec, const std::vector<dnntest::NetworkOp>& ops, unsigned features) {
    dnntest::NetworkTrace trace = dnn::harness::runNetwork<dnntest::feature::All>(spec, ops);
    if ((features & dnntest::feature::NetworkLayerInput) == 0) {
        for (dnntest::NetworkSnapshot& snapshot : trace) {
            snapshot.layerInputs.clear();
        }
    }
    return trace;
}

}
