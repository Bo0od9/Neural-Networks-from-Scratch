#include "CrossEntropyLoss.h"

namespace Network {

double CrossEntropyLoss::computeLoss(const Vector &predicted, const Vector &actual) const {
  assert(predicted.size() == actual.size() && "Predicted and actual vectors must have the same size");
  Vector logPredicted = predicted.array().log();
  double loss = -(actual.array() * logPredicted.array()).sum();
  return loss;
}

Vector CrossEntropyLoss::computeDerivativeLoss(const Vector &predicted, const Vector &actual) const {
  assert(predicted.size() == actual.size() && "Predicted and actual vectors must have the same size");
  return predicted - actual;
}

double CrossEntropyLoss::computeBatchLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const {
  assert(predictedBatch.rows() == actualBatch.rows()
         && "Predicted and actual batches must have the same number of classes");
  assert(predictedBatch.cols() == actualBatch.cols()
         && "Predicted and actual batches must have the same number of samples");
  Matrix logPredictedBatch = predictedBatch.array().log().matrix();
  double loss = -(actualBatch.array() * logPredictedBatch.array()).sum() / predictedBatch.cols();
  return loss;
}

Matrix CrossEntropyLoss::computeBatchDerivativeLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const {
  assert(predictedBatch.rows() == actualBatch.rows()
         && "Predicted and actual batches must have the same number of classes");
  assert(predictedBatch.cols() == actualBatch.cols()
         && "Predicted and actual batches must have the same number of samples");
  return (predictedBatch - actualBatch) / predictedBatch.cols();
}

}// namespace Network
