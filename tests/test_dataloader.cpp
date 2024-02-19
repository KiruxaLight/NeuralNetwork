#include "../DataLoader.h"
#include <iostream>

void PrintImageData(const Eigen::VectorXd& image, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << (image[i * cols + j] > 0.5 ? '1' : '0') << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    try {
        NeuralNetwork::DataLoader loader("./train copy/train-images.idx3-ubyte", "./train copy/train-labels.idx1-ubyte", 4);

        for (size_t batch = 0; batch < 2; ++batch) {
            NeuralNetwork::Batch data;
            loader.NextBatch(data);
            std::cout << "Batch size: " << data.size() << std::endl;
            for (const auto& [image, label] : data) {
                PrintImageData(image, MNIST::ROW_SIZE, MNIST::COLUMN_SIZE);
                std::cout << "Label: " << MNIST::Argmax(label) << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
