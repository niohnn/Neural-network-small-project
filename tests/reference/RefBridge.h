#pragma once

// Interface to the reference implementation. The reference library (libdnn_ref.a) is the full
// implementation compiled with namespace dnn renamed to dnn_ref, so it can be linked next to the
// implementation under test. This header only uses std types and the dnntest types, so it can be
// included together with the headers under test.

#include "HarnessTypes.h"

#include <string>
#include <vector>

namespace dnnref {

using dnntest::Mat;
using dnntest::Vec;

// MathUtils
double dotProduct(const Vec& a, const Vec& b);
double sum(const Vec& values);
Vec normalize(const std::vector<int>& values);
Vec normalize(const std::vector<double>& values);
Vec normalize(const std::vector<int>& values, int maxValue);
Vec normalize(const std::vector<double>& values, int maxValue);

// Activation functions (Sigmoid, ScaledSigmoid, ReLU, LeakyReLU)
double activation(const dnntest::ActSpec& spec, double x);
double activationBackwards(const dnntest::ActSpec& spec, double x);
std::string activationName(const dnntest::ActSpec& spec);
double scaledSigmoidMin(double min, double max);
double scaledSigmoidMax(double min, double max);

// BinaryCrossEntropy
double bceCost(const Vec& predictedOutput, const Vec& actualOutput);
double bceAccuracy(const Vec& predictedOutput, const Vec& actualOutput);
Vec bceGradient(const Vec& predictedOutput, const Vec& actualOutput);

// Layer / Network scripts (see Harness.h). runNetwork's features: the feature mask of the calling test;
// state that the test does not record (getLayerInput() without feature::NetworkLayerInput) is left out.
dnntest::LayerTrace runLayer(const dnntest::LayerSpec& spec, const std::vector<dnntest::LayerOp>& ops);
dnntest::NetworkTrace runNetwork(const dnntest::NetworkSpec& spec, const std::vector<dnntest::NetworkOp>& ops, unsigned features);

}
