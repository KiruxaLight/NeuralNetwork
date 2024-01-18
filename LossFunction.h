#ifndef NEURALNETWORK_LOSSFUNCTION_H
#define NEURALNETWORK_LOSSFUNCTION_H

#include <Eigen/Dense>

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;

    class LossFunction {
    public:
        [[nodiscard]] virtual double Compute(const Vector &predicted,
                                             const Vector &expected) const = 0;

        [[nodiscard]] virtual Vector GetDerivative(const Vector &predicted,
                                                   const Vector &expected) const = 0;
    };

    class MSE final : public LossFunction {
    public:
        [[nodiscard]] double Compute(const Vector &predicted,
                                     const Vector &expected) const final;

        [[nodiscard]] Vector GetDerivative(const Vector &predicted,
                                           const Vector &expected) const final;
    };
} // namespace NeuralNetwork

#endif //NEURALNETWORK_LOSSFUNCTION_H
