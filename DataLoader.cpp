#include "DataLoader.h"

namespace NeuralNetwork {

    DataLoader::DataLoader(const std::string &image_path, const std::string &label_path, int batch_size)
            : batch_size(batch_size), current_index(0) {
        LoadImages(image_path);
        LoadLabels(label_path);
    }

    void DataLoader::NextBatch(Batch& batch) {
        batch.resize(std::min(num_images - current_index, batch_size));
        for (auto& i : batch) {
            i = std::make_pair(LoadImage(), MNIST::ConvertInt(LoadLabel()));
            current_index++;
        }
    }

    void DataLoader::LoadImages(const std::string& path_to_images) {
        load_images = std::ifstream(path_to_images, std::ios::binary);

        if (!load_images.is_open()) {
            throw std::runtime_error("Ошибка при открытии файла");
        }

        uint32_t identity_image_file, num_rows, num_cols;
        load_images.read(reinterpret_cast<char *>(&identity_image_file), MNIST::POINTER);
        load_images.read(reinterpret_cast<char *>(&num_images), MNIST::POINTER);
        load_images.read(reinterpret_cast<char *>(&num_rows), MNIST::POINTER);
        load_images.read(reinterpret_cast<char *>(&num_cols), MNIST::POINTER);

        identity_image_file = __builtin_bswap32(identity_image_file);
        num_images = __builtin_bswap32(num_images);
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);

        size_of_picture = num_cols * num_rows;

        if (size_of_picture != MNIST::IMAGE_SIZE) {
            throw std::runtime_error("Неправильный формат файла");
        }

        if (identity_image_file != MNIST::IDENTITY_IMAGE_FILE) {
            throw std::runtime_error("Неправильный формат изображений файла");
        }
    }

    void DataLoader::LoadLabels(const std::string& path_to_labels) {
        load_labels = std::ifstream(path_to_labels, std::ios::binary);
        if (!load_labels.is_open()) {
            throw std::runtime_error("Ошибка при открытии файла");
        }

        uint32_t identity_label_file = 0, num_items = 0;
        load_labels.read(reinterpret_cast<char *>(&identity_label_file), MNIST::POINTER);
        load_labels.read(reinterpret_cast<char *>(&num_items), MNIST::POINTER);

        identity_label_file = __builtin_bswap32(identity_label_file);

        if (identity_label_file != MNIST::IDENTITY_LABEL_FILE) {
            throw std::runtime_error("Неправильный формат меток файла");
        }
    }

    Eigen::Vector<double, MNIST::IMAGE_SIZE> DataLoader::LoadImage() {
        if (current_index >= num_images) {
            throw std::runtime_error("Индекс выходит за пределы диапазона");
        }

        Vector result(784);

        for (int32_t i = 0; i < size_of_picture; i++) {
            unsigned char temp = 0;
            load_images.read((char *) &temp, sizeof(temp));
            result(i) = temp / MNIST::PIXEL_MAX;
        }
        return result;
    }

    uint8_t DataLoader::LoadLabel() {
        if (current_index >= num_images) {
            throw std::runtime_error("Индекс выходит за пределы диапазона");
        }
        uint8_t label = 0;
        load_labels.read(reinterpret_cast<char *>(&label), 1);
        return label;
    }

    void DataLoader::Reset() {
        current_index = 0;
        load_images.seekg(MNIST::POINTER * 4, std::ios::beg);
        load_labels.seekg(MNIST::POINTER * 2, std::ios::beg);
    }

}; // namespace NeuralNetwork
