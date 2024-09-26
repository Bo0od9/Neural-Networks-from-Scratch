#ifndef CROSSENTROPYLOSS_H_
#define CROSSENTROPYLOSS_H_

#include "LossFunction.h"
#include "declarations.h"

namespace Network {

class CrossEntropyLoss : public LossFunction {
 public:
  double computeLoss(const Vector &predicted, const Vector &actual) const override;
  Vector computeDerivativeLoss(const Vector &predicted, const Vector &actual) const override;
  double computeBatchLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const override;
  Matrix computeBatchDerivativeLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const override;
};

}// namespace Network
#endif// CROSSENTROPYLOSS_H_
