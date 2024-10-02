#ifndef RELUACTIVATION_H_
#define RELUACTIVATION_H_

#include "ActivationFunction.h"
#include "declarations.h"

namespace Network {

class ReLUActivation : public ActivationFunction {
 public:
  Matrix compute(const Matrix &matrix) const override {
    return matrix.cwiseMax(0.0);
  }

  Matrix computeDerivative(const Matrix &matrix) const override {
    return (matrix.array() > 0).cast<double>();
  }
};

}// namespace Network

#endif// RELUACTIVATION_H_
