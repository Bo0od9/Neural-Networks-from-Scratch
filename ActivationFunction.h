#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_
#include <cmath>

namespace Network {
class ActivationFunction {

 public:
  virtual ~ActivationFunction() = default;
  virtual double compute(double x) const = 0;
  virtual double computeDerivative(double x) const = 0;
};
}// namespace Network
#endif//ACTIVATIONFUNCTION_H_
