#ifndef MEANSQUAREDERROR_H_
#define MEANSQUAREDERROR_H_

#include "LossFunction.h"
#include "declarations.h"

namespace Network {

class MeanSquaredError : public LossFunction {
 public:
  double computeLoss(const Vector &predicted, const Vector &actual) const override;
  Vector computeDerivativeLoss(const Vector &predicted, const Vector &actual) const override;
  double computeBatchLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const override;
  Matrix computeBatchDerivativeLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const override;
};

}// namespace Network
#endif//MEANSQUAREDERROR_H_
