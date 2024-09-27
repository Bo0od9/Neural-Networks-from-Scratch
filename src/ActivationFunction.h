#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_
#include "declarations.h"

namespace Network {

class ActivationFunction {
 public:
  virtual ~ActivationFunction() = default;
  virtual Matrix compute(const Matrix &matrix) const = 0;
  virtual Matrix computeDerivative(const Matrix &matrix) const = 0;
};

}// namespace Network
#endif//ACTIVATIONFUNCTION_H_
