#ifndef NEURALNETWORK_SRC_CONSTS_H_
#define NEURALNETWORK_SRC_CONSTS_H_

#endif //NEURALNETWORK_SRC_CONSTS_H_

#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace MNIST {

    const int32_t ROW_SIZE = 28;
    const int32_t COLUMN_SIZE = 28;
    const int32_t IMAGE_SIZE = 784;
    const int32_t COUNT_OF_DIGITS = 10;
    const std::streamsize POINTER = 4;
    const double PIXEL_MAX = 255.0;
    const int32_t IDENTITY_IMAGE_FILE = 0x00000803;
    const int32_t IDENTITY_LABEL_FILE = 2049;

    static Eigen::VectorXd ConvertInt(int32_t number) {
        Eigen::VectorXd y = Eigen::Vector<double, MNIST::COUNT_OF_DIGITS>();
        y.setZero();
        y[number] = 1;
        return y;
    }

    static int32_t Argmax(const Eigen::VectorXd &y) {
        int32_t index = 0;
        for (int32_t i = 0; i < MNIST::COUNT_OF_DIGITS; ++i) {
            if (y[i] > y[index]) {
                index = i;
            }
        }
        return index;
    }

}