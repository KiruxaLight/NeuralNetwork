#ifndef NEURALNETWORK_SRC_DATALOADER_H_
#define NEURALNETWORK_SRC_DATALOADER_H_

#include "MNIST.h"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace NeuralNetwork {

    using Vector = Eigen::VectorXd;
    using Batch = std::vector<std::pair<Vector, Vector>>;

    class DataLoader {
    public:
        DataLoader(const std::string &image_path, const std::string &label_path, int batch_size);

        void NextBatch(Batch& batch);
        void Reset();

    private:
        std::ifstream load_images;
        std::ifstream load_labels;

        size_t batch_size;
        size_t current_index = 0;
        size_t num_images;
        size_t size_of_picture;

        void LoadImages(const std::string &image_path);
        void LoadLabels(const std::string &label_path);

        Eigen::Vector<double, MNIST::IMAGE_SIZE> LoadImage();
        uint8_t LoadLabel();
    };
}; // namespace NeuralNetwork

#endif // NEURALNETWORK_SRC_DATALOADER_H_
