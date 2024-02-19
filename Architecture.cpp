//
// Created by Kirill on 16.01.2024.
//

#include "Architecture.h"

namespace NeuralNetwork {

    Architecture::Architecture(std::initializer_list<int16_t> sizes,
                               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions)
            : activation_functions_(std::move(activation_functions)) {

        size_t layers_count = activation_functions_.size();

        assert(sizes.size() == layers_count + 1);

        layers_.reserve(layers_count);
        for (auto it = sizes.begin(); it + 1 != sizes.end(); ++it) {
            layers_.push_back(LinearLayer(*(it + 1), *it));
        }

        a_grad_.resize(layers_count);
        b_grad_.resize(layers_count);
        current_x_.resize(layers_count);
        current_y_.resize(layers_count);

        for (size_t layer = 0; layer < layers_count; ++layer) {
            a_grad_[layer] = layers_[layer].a;
            b_grad_[layer] = layers_[layer].b;
        }

        ResetGrad();
    }

    void Architecture::Compute(Vector &x) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_x_[i] = x;
            current_y_[i] = activation_functions_[i]->Compute(layers_[i].Compute(x));
            x = current_y_[i];
        }
    }

    void Architecture::BackPropagate(Vector &x) {
        for (int32_t i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            Vector activ_x = activation_functions_[i]->GetDerivative(current_y_[i]) * x;
            a_grad_[i] += activ_x * current_x_[i].transpose();
            b_grad_[i] += activ_x;
            x = layers_[i].a.transpose() * activ_x;
        }
    }

    void Architecture::Step(double learning_rate) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].Step(a_grad_[i], b_grad_[i], learning_rate);
        }
    }

    void Architecture::ResetGrad() {
        for (size_t i = 0; i < layers_.size(); ++i) {
            a_grad_[i].setZero();
            b_grad_[i].setZero();
        }
    }
} // namespace NeuralNetwork