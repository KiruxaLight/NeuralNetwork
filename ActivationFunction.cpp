//
// Created by Kirill on 16.01.2024.
//

#include "ActivationFunction.h"

namespace NeuralNetwork {
    [[nodiscard]] Vector ReLU::Compute(const Vector &x) const {
        return x.cwiseMax(0.0);
    }

    [[nodiscard]] Matrix ReLU::GetDerivative(const Vector &x) const {
        return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
    }

    [[nodiscard]] Vector Softmax::Compute(const NeuralNetwork::Vector &x) const {
        auto result = x.array().exp();
        return result / result.sum();
    }

    [[nodiscard]] Matrix Softmax::GetDerivative(const NeuralNetwork::Vector &x) const {
        Vector computeSoftmax = Compute(x);
        Matrix diagonal = computeSoftmax.asDiagonal();
        return diagonal - computeSoftmax * computeSoftmax.transpose();
    }
} // namespace NeuralNetwork