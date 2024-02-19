//
// Created by Kirill on 16.01.2024.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include "Eigen/Dense"

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    class LinearLayer {
    public:
        LinearLayer(int16_t n, int16_t m);

        [[nodiscard]] Vector Get(const Vector &x) const;
        void Step(const Matrix &da, const Vector &db, double learning_rate);
        Vector Compute(const Vector &x) const;
    private:
        friend class Architecture;
        Matrix a;
        Vector b;
    };

} // namespace NeuralNetwork


#endif //NEURALNETWORK_LAYER_H
