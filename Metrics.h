//
// Created by Kirill on 18.01.2024.
//

#ifndef NEURALNETWORK_METRICS_H
#define NEURALNETWORK_METRICS_H

#include "Eigen/Dense"

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;

    double GetAccuracy(const std::vector<Vector> &predicted, const std::vector<Vector> &expected);

} // namespace NeuralNetwork


#endif //NEURALNETWORK_METRICS_H
