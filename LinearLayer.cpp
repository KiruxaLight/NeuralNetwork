//
// Created by Kirill on 16.01.2024.
//

#include "LinearLayer.h"

namespace NeuralNetwork {

    LinearLayer::LinearLayer(int16_t n, int16_t m) {
        a = Matrix::Random(n, m);
        b = Vector::Random(n);
    }

    Vector LinearLayer::Get(const Vector &x) const {
        return a * x + b;
    }

    void LinearLayer::Step(const Matrix &da, const Vector &db, double learning_rate) {
        a -= da * learning_rate;
        b -= db * learning_rate;
    }

    Vector LinearLayer::Compute(const Vector &x) const {
        return a * x + b;
    }
} // namespace NeuralNetwork
