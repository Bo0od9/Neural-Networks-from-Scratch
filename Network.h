#ifndef NETWORK_H_
#define NETWORK_H_

#include "Layer.h"
#include <vector>

namespace Network {

class NeuralNetwork {
 public:
  NeuralNetwork(const ActivationFunctionPtr &activationFunction,
                const std::vector<size_t> &neuronsPerLayer);

 private:
  std::vector<Layer> layers_;
  Vector forwardPropagation(const Vector &input);
  void backwardPropagation(const Vector &dLoss_dOutput, double learningRate);
};

}// namespace Network
#endif//NETWORK_H_
