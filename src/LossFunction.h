#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_

#include "declarations.h"

namespace Network {

class LossFunction {
 public:
  virtual ~LossFunction() = default;
  virtual double computeLoss(const Vector &predicted, const Vector &actual) const = 0;
  virtual Vector computeDerivativeLoss(const Vector &predicted, const Vector &actual) const = 0;
  virtual double computeBatchLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const = 0;
  virtual Matrix computeBatchDerivativeLoss(const Matrix &predictedBatch, const Matrix &actualBatch) const = 0;
};

}// namespace Network
#endif//LOSSFUNCTION_H_
