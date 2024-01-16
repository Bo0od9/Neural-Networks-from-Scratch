#ifndef LAYER_H_
#define LAYER_H_
#include "SigmoidActivation.h"
#include <Eigen/Dense>
#include <memory>
#include <random>

namespace Network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Size = size_t;
using ActivationFunctionPtr = std::unique_ptr<ActivationFunction>;

class Layer {

 public:
  Layer(Size outputSize, Size inputSize, ActivationFunctionPtr activationFunction);

  //These methods are intended for debugging purposes
  [[nodiscard]] const Matrix &getWeights() const { return weights_; }
  [[nodiscard]] const Vector &getBiases() const { return biases_; }
  [[nodiscard]] const Vector &getOutput() const { return output_; }
  void setWeight(const Matrix &input);
  void setBiases(const Vector &input);

  Vector forward(const Vector &input);
  Vector backward(const Vector &dLoss_dOutput);
  void updateWeights(double learningRate);

 private:
  Size inputSize_;
  Size outputSize_;
  Matrix weights_;
  Vector biases_;
  Vector input_;
  Vector output_;
  Matrix dLoss_dWeights_;
  Vector dLoss_dBiases_;

  ActivationFunctionPtr activationFunction_;
};
}// namespace Network
#endif
