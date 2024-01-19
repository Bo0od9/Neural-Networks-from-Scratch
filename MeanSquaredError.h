#ifndef MSELOSSFUNCTION_H_
#define MSELOSSFUNCTION_H_
#include "LossFunction.h"

namespace Network {
class MeanSquaredError : public LossFunction {
  inline double computeLoss(const Vector &predicted, const Vector &actual) const override {
    return (actual - predicted).squaredNorm() / actual.size();
  }
  inline Vector computeDerivativeLoss(const Vector &predicted, const Vector &actual) const override {
    return 2.0 * (predicted - actual) / actual.size();
  }
};
}// namespace Network
#endif//MSELOSSFUNCTION_H_
