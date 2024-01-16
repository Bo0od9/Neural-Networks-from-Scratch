#include "Layer.h"

namespace Network {
Layer::Layer(Size outputSize, Size inputSize, ActivationFunctionPtr activationFunction) : inputSize_(inputSize), outputSize_(outputSize), activationFunction_(std::move(activationFunction)) {
  //Uses normal distribution to generate weights and biases
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 1);

  weights_ = Matrix(outputSize, inputSize).unaryExpr([&](double _) { return d(gen); });
  biases_ = Vector(outputSize).unaryExpr([&](double _) { return d(gen); });
}
Vector Layer::forward(const Network::Vector &input) {
  input_ = input;
  output_ = (weights_ * input + biases_).unaryExpr([this](double x) { return activationFunction_->compute(x); });
  return output_;
}

Vector Layer::backward(const Network::Vector &dLoss_dOutput) {
  Vector activatedDerivative = (weights_ * input_ + biases_).unaryExpr([this](double x) { return activationFunction_->computeDerivative(x); });
  Vector errorGradientProduct = dLoss_dOutput.cwiseProduct(activatedDerivative);
  dLoss_dWeights_ = errorGradientProduct * input_.transpose();
  dLoss_dBiases_ = errorGradientProduct;
  Vector inputGradient = weights_.transpose() * errorGradientProduct;
  return inputGradient;
}
void Layer::updateWeights(double learningRate) {
  weights_ -= dLoss_dWeights_ * learningRate;
  biases_ -= dLoss_dBiases_ * learningRate;
}
void Layer::setWeight(const Network::Matrix &input) {
  assert(input.innerSize() == outputSize_ && input.outerSize() == inputSize_);
  weights_ = input;
}
void Layer::setBiases(const Network::Vector &input) {
  assert(input.size() == outputSize_);
  biases_ = input;
}
}// namespace Network