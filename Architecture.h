//
// Created by Kirill on 16.01.2024.
//

#ifndef NEURALNETWORK_ARCHITECTURE_H
#define NEURALNETWORK_ARCHITECTURE_H

#include "Eigen/Dense"
#include "LinearLayer.h"
#include "LossFunction.h"
#include "ActivationFunction.h"
#include "DataLoader.h"

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    class Architecture {
    public:
        Architecture(std::initializer_list<int16_t> sizes,
                     std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions);

    private:
        friend class Model;
        std::vector<LinearLayer> layers_;
        std::vector<std::unique_ptr<ActivationFunction>> activation_functions_;
        std::vector<Matrix> a_grad_;
        std::vector<Vector> b_grad_;
        std::vector<Vector> current_x_;
        std::vector<Vector> current_y_;

        void ResetGrad();
        void Compute(Vector &x);
        void BackPropagate(Vector &x);
        void Step(double learning_rate);
    };

}; // namespace NeuralNetwork

#endif //NEURALNETWORK_ARCHITECTURE_H
