#include "MNISTLoader.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace Data {

MNISTLoader::MNISTLoader(const std::string &imagesPath, const std::string &labelsPath)
    : imagesPath_(imagesPath),
      labelsPath_(labelsPath),
      numImages_(0),
      numRows_(0),
      numCols_(0) {}
bool MNISTLoader::loadData() {
  return loadImages() && loadLabels();
}

bool MNISTLoader::loadImages() {
  std::ifstream imagesFile(imagesPath_, std::ios::binary);
  if (!imagesFile.is_open()) {
    std::cerr << "Не удалось открыть файл: " << imagesPath_ << std::endl;
    return false;
  }
  uint32_t magicNumber = readUint32(imagesFile);
  if (magicNumber != 2051) {
    std::cerr << "Неверный magic number в файле: " << magicNumber << std::endl;
    return false;
  }
  numImages_ = readUint32(imagesFile);
  numRows_ = readUint32(imagesFile);
  numCols_ = readUint32(imagesFile);
  images_.reserve(numImages_);
  for (uint32_t i = 0; i < numImages_; ++i) {
    Matrix image(numRows_, numCols_);
    for (uint32_t r = 0; r < numRows_; ++r) {
      for (uint32_t c = 0; c < numCols_; ++c) {
        uint8_t pixel = 0;
        imagesFile.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
        image(r, c) = static_cast<double>(pixel) / 255.0;
      }
    }
    images_.push_back(std::move(image));
  }
  return true;
}

bool MNISTLoader::loadLabels() {
  std::ifstream labelsFile(labelsPath_, std::ios::binary);
  if (!labelsFile.is_open()) {
    std::cerr << "Не удалось открыть файл меток: " << labelsPath_ << std::endl;
    return false;
  }
  uint32_t magicNumber = readUint32(labelsFile);
  if (magicNumber != 2049) {
    std::cerr << "Неверный magic number в файле меток: " << magicNumber << std::endl;
    return false;
  }
  uint32_t numLabels = readUint32(labelsFile);
  if (numLabels != numImages_) {
    std::cerr << "Количество меток не совпадает с количеством изображений." << std::endl;
    return false;
  }
  labels_.reserve(numLabels);
  for (uint32_t i = 0; i < numLabels; ++i) {
    uint8_t label = 0;
    labelsFile.read(reinterpret_cast<char *>(&label), sizeof(label));
    labels_.push_back(label);
  }
  return true;
}

uint32_t MNISTLoader::readUint32(std::ifstream &stream) {
  uint32_t value = 0;
  stream.read(reinterpret_cast<char *>(&value), sizeof(value));
  return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | ((value & 0xFF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

std::vector<std::pair<Vector, Vector>> MNISTLoader::getDataset() const {
  std::vector<std::pair<Vector, Vector>> dataset;
  dataset.reserve(images_.size());
  for (size_t i = 0; i < images_.size(); ++i) {
    Vector imageVector = Eigen::Map<const Vector>(images_[i].data(), images_[i].size());
    Vector labelVector = Vector::Zero(10);
    labelVector(labels_[i]) = 1.0;
    dataset.emplace_back(std::move(imageVector), std::move(labelVector));
  }
  return dataset;
}

}// namespace Data
