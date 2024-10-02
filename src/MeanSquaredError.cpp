#include "MeanSquaredError.h"

namespace Network {

double MeanSquaredError::computeLoss(const Vector &predicted, const Vector &actual) const {
  assert(predicted.size() == actual.size() && "Predicted and actual vectors must be of the same size");
  return (predicted - actual).squaredNorm() / actual.size();
}

Vector MeanSquaredError::computeDerivativeLoss(const Vector &predicted, const Vector &actual) const {
  assert(predicted.size() == actual.size() && "Predicted and actual vectors must be of the same size");
  return 2.0 * (predicted - actual) / actual.size();
}

double MeanSquaredError::computeBatchLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const {
  assert(predictedBatch.rows() == actualBatch.rows()
         && "Predicted and actual batches must have the same output size");
  assert(predictedBatch.cols() == actualBatch.cols()
         && "Predicted and actual batches must have the same number of samples");
  Matrix diff = predictedBatch - actualBatch;
  return diff.squaredNorm() / predictedBatch.cols();
}

Matrix MeanSquaredError::computeBatchDerivativeLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const {
  assert(predictedBatch.rows() == actualBatch.rows()
         && "Predicted and actual batches must have the same output size");
  assert(predictedBatch.cols() == actualBatch.cols()
         && "Predicted and actual batches must have the same number of samples");
  return 2.0 * (predictedBatch - actualBatch) / predictedBatch.cols();
}

}// namespace Network