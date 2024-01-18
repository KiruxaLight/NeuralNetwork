//
// Created by Kirill on 16.01.2024.
//

#ifndef NEURALNETWORK_ACTIVATIONFUNCTIONS_H
#define NEURALNETWORK_ACTIVATIONFUNCTIONS_H

#include "Eigen/Dense"

namespace NeuralNetwork {
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    class ActivationFunction {
    public:
        [[nodiscard]] virtual Vector Compute(const Vector &x) const = 0;
        [[nodiscard]] virtual Matrix GetDerivative(const Vector &x) const = 0;
    };

    class ReLU final : public ActivationFunction {
    public:
        [[nodiscard]] Vector Compute(const Vector &x) const final;
        [[nodiscard]] Matrix GetDerivative(const Vector &x) const final;
    };

    class Softmax : public ActivationFunction {
    public:
        [[nodiscard]] Vector Compute(const Vector &x) const final;
        [[nodiscard]] Matrix GetDerivative(const Vector &x) const final;
    };

} // namespace NeuralNetwork


#endif //NEURALNETWORK_ACTIVATIONFUNCTIONS_H
