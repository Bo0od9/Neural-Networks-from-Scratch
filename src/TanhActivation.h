#ifndef TANHACTIVATION_H_
#define TANHACTIVATION_H_

#include "ActivationFunction.h"
#include "declarations.h"

namespace Network {

class TanhActivation : public ActivationFunction {
 public:
  Matrix compute(const Matrix &matrix) const override {
    return matrix.array().tanh();
  }

  Matrix computeDerivative(const Matrix &matrix) const override {
    return 1.0 - compute(matrix).array().square();
  }
};

}// namespace Network
#endif//TANHACTIVATION_H_
