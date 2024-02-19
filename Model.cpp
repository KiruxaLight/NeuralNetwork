//
// Created by Kirill on 14.01.2024.
//

#include "Model.h"

namespace NeuralNetwork {
    Model::Model(Architecture architecture, std::unique_ptr<LossFunction> loss_function, double learning_rate)
            : architecture_(std::move(architecture)), loss_function_(std::move(loss_function)),
              learning_rate_(learning_rate) {}

    void Model::Train(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size,
                      size_t epochs) {
        DataLoader data_loader_train(path_to_images, path_to_labels, batch_size);
        Batch batch;

        std::cout << "Train..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "Current epoch: " << epoch << std::endl;
            data_loader_train.NextBatch(batch);
            int ind = 0;
            while (!batch.empty()) {
                if (ind == NUM_OF_TRAINING) {
                    break;
                }
                if (ind % 1000 == 0) {
                    std::cout << "Current batch: " << ind << std::endl;
                }
                ++ind;
                BackForwardPropagate(batch);
                data_loader_train.NextBatch(batch);
            }
            data_loader_train.Reset();
        }
    }

    std::vector<Vector> Model::GetTestData(const std::string &path_to_images, const std::string &path_to_labels) {
        DataLoader test_data(path_to_images, path_to_labels, 1);
        test_data.Reset();
        Batch batch;
        test_data.NextBatch(batch);

        std::vector<Vector> ans;
        int ind = 0;
        std::cout << "GetTestData..." << std::endl;
        while (!batch.empty()) {
            if (ind == NUM_OF_TESTING) {
                break;
            }
            ++ind;
            for (auto &[predict, result]: batch) {
                ans.push_back(result);
            }
            test_data.NextBatch(batch);
        }
        test_data.Reset();
        return ans;
    }

    std::vector<Vector>
    Model::Predict(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size) {
        DataLoader test_data(path_to_images, path_to_labels, batch_size);
        test_data.Reset();
        size_t count_right_answers = 0;
        size_t count_all_images = 0;
        Batch batch;
        test_data.NextBatch(batch);

        std::vector<Vector> ans;
        std::cout << "GetPredictData..." << std::endl;
        int ind = 0;
        while (!batch.empty()) {
            Conversion(batch);
            for (auto &[predict, result]: batch) {
                if (ind == NUM_OF_TESTING) {
                    break;
                }
                ++ind;
                ans.push_back(predict);
                if (MNIST::Argmax(result) == MNIST::Argmax(predict)) {
                    ++count_right_answers;
                }
            }
            count_all_images += batch.size();
            test_data.NextBatch(batch);
        }
        test_data.Reset();
        return ans;
    }

    void Model::BackForwardPropagate(NeuralNetwork::Batch &batch) {
        architecture_.ResetGrad();
        Conversion(batch);
        BackPropagate(batch);
        architecture_.Step(learning_rate_);
    }

    void Model::Conversion(NeuralNetwork::Batch &batch) {
        for (auto &[input, result]: batch) {
            architecture_.Compute(input);
        }
    }

    void Model::BackPropagate(const Batch &batch) {
        Vector derivative(batch[0].first.size());
        derivative.setZero();
        for (auto &[predict, result]: batch) {
            derivative += loss_function_->GetDerivative(predict, result);
        }
        architecture_.BackPropagate(derivative);
    }

} // namespace NeuralNetwork