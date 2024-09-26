#ifndef NETWORK_H_
#define NETWORK_H_

#include "Layer.h"
#include "LossFunction.h"
#include <iostream>
#include <vector>

namespace Network {

class NeuralNetwork {
 public:
  explicit NeuralNetwork(const LossFunction *lossFunction) : lossFunction_(lossFunction) {}
  void addLayer(const Layer &layer);
  void train(const std::vector<Vector> &inputs, const std::vector<Vector> &targets,
             int epochs, double learningRate, int batchSize);
  Matrix predict(const Matrix &inputs);

 private:
  std::vector<Layer> layers_;
  const LossFunction *lossFunction_;
};

}// namespace Network

#endif// NETWORK_H_
