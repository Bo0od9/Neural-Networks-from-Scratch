#ifndef MSELOSSFUNCTION_H_
#define MSELOSSFUNCTION_H_
#include "LossFunction.h"

namespace Network {
class MeanSquaredError : public LossFunction {
  double computeLoss(const Vector &actual, const Vector &predicted) const override {
    return (actual - predicted).squaredNorm() / actual.size();
  }
  Vector computeDerivativeLoss(const Vector &actual, const Vector &predicted) const override {
    return 2.0 * (predicted - actual) / actual.size();
  }
};
}// namespace Network
#endif//MSELOSSFUNCTION_H_
