#ifndef LAYER_H_
#define LAYER_H_

#include "ActivationFunction.h"
#include "EigenRand/EigenRand"
#include "declarations.h"

namespace Network {

class Layer {
 public:
  Layer(Index inputSize, Index outputSize, ActivationFunction &activationFunction);
  Matrix forward(const Matrix &input);
  Matrix backward(const Matrix &gradOutput);
  void updateWeights(double learningRate, Index batchSize);
  const Matrix &getWeights() const { return weights_; }
  const Vector &getBiases() const { return biases_; }
  Index getOutputSize() const { return outputSize_; }
  Index getInputSize() const { return inputSize_; }

 private:
  void initializeWeightsAndBiases();
  void zeroGradients();
  ActivationFunction &activationFunction_;
  const Index inputSize_;
  const Index outputSize_;
  Vector grad_biases_;
  Matrix grad_weights_;
  Matrix weights_;
  Vector biases_;
  Matrix input_;
  Matrix output_;
};

}// namespace Network

#endif// LAYER_H_
