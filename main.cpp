#include <iostream>
#include <Eigen/Dense>
#include "Model.h"
#include "ActivationFunction.h"
#include "Metrics.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

int main() {
    std::vector<std::unique_ptr<NeuralNetwork::ActivationFunction> > activation_functions;
    activation_functions.push_back(std::make_unique<NeuralNetwork::ReLU>());
//    activation_functions.push_back(std::make_unique<NeuralNetwork::ReLU>());
    activation_functions.push_back(std::make_unique<NeuralNetwork::Softmax>());
    std::unique_ptr<NeuralNetwork::LossFunction> loss_function = std::make_unique<NeuralNetwork::MSE>();
    double learning_rate = 0.1;
    NeuralNetwork::Model model(NeuralNetwork::Architecture({784, 128, 10}, std::move(activation_functions)),
                               std::move(loss_function), learning_rate);
    model.Train("../train copy/train-images.idx3-ubyte", "../train copy/train-labels.idx1-ubyte", 4, 2);
    std::vector<Vector> predicted = model.Predict("../train copy/train-images.idx3-ubyte", "../train copy/train-labels.idx1-ubyte", 4);
    std::vector<Vector> excepted = model.GetTestData("../train copy/train-images.idx3-ubyte", "../train copy/train-labels.idx1-ubyte");
    std::cout << "Accuracy: " << NeuralNetwork::GetAccuracy(predicted, excepted) << "\n";
}
