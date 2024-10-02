#include "Network.h"

namespace Network {

void NeuralNetwork::addLayer(const Layer &layer) {
  assert(layer.getInputSize() > 0 && layer.getOutputSize() > 0 && "Layer must have valid input and output sizes");
  if (!layers_.empty()) {
    assert(layers_.back().getOutputSize() == layer.getInputSize() && "Layer sizes do not match");
  }
  layers_.push_back(layer);
}

void NeuralNetwork::train(const std::vector<Vector> &inputs, const std::vector<Vector> &targets,
                          int epochs, double learningRate, int batchSize) {
  assert(inputs.size() == targets.size() && "The number of inputs must match the number of target values");
  assert(epochs > 0 && "Number of epochs must be greater than 0");
  assert(learningRate > 0 && "Learning rate must be greater than 0");
  assert(batchSize > 0 && "Batch size must be greater than 0");
  assert(!layers_.empty() && "Neural network must contain at least one layer");
  assert(lossFunction_ != nullptr && "Loss function must be set");
  assert(!inputs.empty() && "Inputs should not be empty");
  assert(!targets.empty() && "Targets should not be empty");

  Index numSamples = inputs.size();
  for (size_t epoch = 1; epoch <= epochs; ++epoch) {
    double epochLoss = 0.0;

    for (Index i = 0; i < numSamples; i += batchSize) {
      Index currentBatchSize = std::min(batchSize, static_cast<int>(numSamples - i));
      assert(currentBatchSize > 0 && "The size of the current batch must be greater than 0");
      Matrix inputBatch(inputs[0].size(), currentBatchSize);
      Matrix targetBatch(targets[0].size(), currentBatchSize);

      for (Index j = 0; j < currentBatchSize; ++j) {
        assert(inputs[i + j].size() == inputs[0].size() && "Input vectors must have consistent size");
        assert(targets[i + j].size() == targets[0].size() && "Target vectors must have consistent size");
        inputBatch.col(j) = inputs[i + j];
        targetBatch.col(j) = targets[i + j];
      }
      Matrix outputBatch = inputBatch;
      for (auto &layer : layers_) {
        outputBatch = layer.forward(outputBatch);
      }
      double batchLoss = lossFunction_->computeBatchLoss(outputBatch, targetBatch);
      assert(std::isfinite(batchLoss) && "Loss value must be finite");
      epochLoss += batchLoss * currentBatchSize;
      Matrix lossGradients = lossFunction_->computeBatchDerivativeLoss(outputBatch, targetBatch);
      Matrix grad = lossGradients;

      for (int k = static_cast<int>(layers_.size()) - 1; k >= 0; --k) {
        grad = layers_[k].backward(grad);
        assert(grad.allFinite() && "Gradients must be finite values");
      }
      for (auto &layer : layers_) {
        layer.updateWeights(learningRate, currentBatchSize);
      }
    }
    double averageLoss = epochLoss / numSamples;
    assert(std::isfinite(averageLoss) && "Average loss value must be finite");
    std::cout << "Epoch " << epoch << " - Loss: " << averageLoss << std::endl;
  }
}

Vector NeuralNetwork::predict(const Vector &inputs) {
  assert(inputs.rows() == layers_.front().getInputSize() && "Input data size must match the input size of the first layer");
  Matrix output = inputs;
  for (auto &layer : layers_) {
    output = layer.forward(output);
  }
  return output;
}

}// namespace Network
