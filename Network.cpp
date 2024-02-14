#include "Network.h"

namespace Network {
NeuralNetwork::NeuralNetwork(const ActivationFunctionPtr &activationFunction,
                             const std::vector<size_t> &neuronsPerLayer) {

  Size previousLayerSize = neuronsPerLayer.front();
  for (size_t i = 1; i < neuronsPerLayer.size(); ++i) {
    Size currentLayerSize = neuronsPerLayer[i];
    Layer layer(currentLayerSize, previousLayerSize, activationFunction);

    layers_.push_back(layer);
    previousLayerSize = currentLayerSize;
  }
}
Vector NeuralNetwork::forwardPropagation(const Network::Vector &input) {
  Vector currentInput = input;
  for (auto &layer : layers_) {
    currentInput = layer.forward(currentInput);
  }
  return currentInput;
}
void NeuralNetwork::backwardPropagation(const Vector &dLoss_dOutput, double learningRate) {
  Vector currentGradient = dLoss_dOutput;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    currentGradient = it->backward(currentGradient);
    it->updateWeights(learningRate);
  }
}

}// namespace Network