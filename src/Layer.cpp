#include "Layer.h"
#include <random>

namespace Network {

Layer::Layer(Index inputSize, Index outputSize, ActivationFunction &activationFunction)
    : inputSize_(inputSize),
      outputSize_(outputSize),
      grad_biases_(Vector::Zero(outputSize)),
      grad_weights_(Matrix::Zero(outputSize, inputSize)),
      weights_(Matrix::Zero(outputSize, inputSize)),
      biases_(Vector::Zero(outputSize)),
      activationFunction_(activationFunction) {
  assert(inputSize > 0 && "Input size must be greater than 0");
  assert(outputSize > 0 && "Output size must be greater than 0");
  initializeWeightsAndBiases();
}

void Layer::initializeWeightsAndBiases() {
  std::random_device rd;
  Eigen::Rand::Vmt19937_64 urng{rd()};
  Eigen::Rand::NormalGen<double> norm_gen{0, 1};
  weights_ = norm_gen.generate<Matrix>(outputSize_, inputSize_, urng) * std::sqrt(2.0 / inputSize_);
  biases_ = Vector::Zero(outputSize_);
}

Matrix Layer::forward(const Matrix &input) {
  assert(input.rows() == inputSize_ && "Input matrix must have the correct number of rows");
  input_ = input;
  Matrix z = (weights_ * input).colwise() + biases_;
  output_ = activationFunction_.compute(z);
  assert(output_.rows() == outputSize_ && "Output matrix must have the correct number of rows");
  return output_;
}

Matrix Layer::backward(const Matrix &gradOutput) {
  assert(gradOutput.rows() == outputSize_ && "Gradient output must have the correct number of rows");
  Matrix z = (weights_ * input_).colwise() + biases_;
  Matrix delta = activationFunction_.computeDerivative(z).array() * gradOutput.array();
  grad_biases_ += delta.rowwise().sum();
  grad_weights_ += delta * input_.transpose();
  // Проверка корректности обновлённых градиентов
  assert(grad_biases_.allFinite() && "Gradient biases must not contain NaN or infinite values");
  assert(grad_weights_.allFinite() && "Gradient weights must not contain NaN or infinite values");
  Matrix gradInput = weights_.transpose() * delta;
  return gradInput;
}

void Layer::updateWeights(double learningRate, Index batchSize) {
  assert(batchSize > 0 && "Batch size must be greater than 0");
  assert(learningRate > 0 && "Learning rate must be greater than 0");
  weights_ -= (learningRate / batchSize) * grad_weights_;
  biases_ -= (learningRate / batchSize) * grad_biases_;
  // Проверка корректности обновлённых весов
  assert(weights_.allFinite() && "Updated weights must not contain NaN or infinite values");
  assert(biases_.allFinite() && "Updated biases must not contain NaN or infinite values");
  zeroGradients();
}

void Layer::zeroGradients() {
  grad_weights_.setZero();
  grad_biases_.setZero();
}

}// namespace Network
