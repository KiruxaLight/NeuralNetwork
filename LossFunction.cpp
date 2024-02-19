//
// Created by Kirill on 14.01.2024.
//

#include "LossFunction.h"

namespace NeuralNetwork {
        double MSE::Compute(const Vector &predicted, const Vector &expected) const {
            return (expected - predicted).squaredNorm() / expected.size();
        }

        Vector MSE::GetDerivative(const Vector &predicted, const Vector &expected) const {
            return 2 * (predicted - expected) / expected.size();
        }
} // namespace NeuralNetwork