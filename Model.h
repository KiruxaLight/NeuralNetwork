//
// Created by Kirill on 14.01.2024.
//

#ifndef NEURALNETWORK_MODEL_H
#define NEURALNETWORK_MODEL_H

#include "Architecture.h"
#include "LinearLayer.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include "Eigen/Dense"

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;

    class Model {
    public:
        Model(Architecture architecture, std::unique_ptr<LossFunction> loss_function, double learning_rate);

        void
        Train(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size, size_t epochs);

        std::vector<Vector>
        Predict(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size);

        std::vector<Vector> GetTestData(const std::string &path_to_images, const std::string &path_to_labels);

    private:
        Architecture architecture_;
        std::unique_ptr<LossFunction> loss_function_;
        double learning_rate_;

        const size_t NUM_OF_TRAINING = 50000;
        const size_t NUM_OF_TESTING = 10000;

        void BackForwardPropagate(Batch &batch);

        void Conversion(Batch &batch);

        void BackPropagate(const Batch &batch);
    };

} // namespace NeuralNetwork


#endif //NEURALNETWORK_MODEL_H
