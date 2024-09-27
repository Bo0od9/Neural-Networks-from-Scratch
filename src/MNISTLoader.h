#ifndef MNISTLOADER_H_
#define MNISTLOADER_H_

#include "declarations.h"
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace Data {

class MNISTLoader {
 public:
  MNISTLoader(const std::string &imagesPath, const std::string &labelsPath);
  bool loadData();
  const std::vector<Matrix> &getImages() const { return images_; };
  const std::vector<uint8_t> &getLabels() const { return labels_; };
  std::vector<std::pair<Vector, Vector>> getDataset() const;

 private:
  uint32_t readUint32(std::ifstream &stream);
  bool loadImages();
  bool loadLabels();
  std::string imagesPath_;
  std::string labelsPath_;
  std::vector<Matrix> images_;
  std::vector<uint8_t> labels_;
  uint32_t numImages_;
  uint32_t numRows_;
  uint32_t numCols_;
};

}// namespace Data

#endif// MNISTLOADER_H_
