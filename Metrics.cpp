//
// Created by Kirill on 18.01.2024.
//

#include "Metrics.h"
#include "MNIST.h"

namespace NeuralNetwork {

    double GetAccuracy(const std::vector<Vector> &predicted, const std::vector<Vector> &expected) {
        assert(expected.size() == predicted.size());

        size_t correct = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (MNIST::Argmax(predicted[i]) == MNIST::Argmax(expected[i])) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / expected.size();
    }
} // namespace NeuralNetwork
