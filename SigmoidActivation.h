#ifndef SIGMOIDACTIVATION_H_
#define SIGMOIDACTIVATION_H_

#include "ActivationFunction.h"
#include "declarations.h"

namespace Network {

class SigmoidActivation : public ActivationFunction {
 public:
  Matrix compute(const Matrix &matrix) const override {
    return (1.0 + (-matrix.array()).exp()).inverse();
  }

  Matrix computeDerivative(const Matrix &matrix) const override {
    Matrix sigmoidMatrix = compute(matrix);
    return sigmoidMatrix.array() * (1 - sigmoidMatrix.array());
  }
};

}// namespace Network
#endif//SIGMOIDACTIVATION_H_
