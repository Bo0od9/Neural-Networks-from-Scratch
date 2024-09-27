#ifndef SOFTMAXACTIVATION_H_
#define SOFTMAXACTIVATION_H_

#include "ActivationFunction.h"
#include "declarations.h"

namespace Network {

class SoftmaxActivation : public ActivationFunction {
 public:
  Matrix compute(const Matrix &matrix) const override;
  Matrix computeDerivative(const Matrix &matrix) const override;
};

}// namespace Network
#endif// SOFTMAXACTIVATION_H_