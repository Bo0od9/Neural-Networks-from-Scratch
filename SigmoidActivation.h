#ifndef SIGMOIDACTIVATION_H_
#define SIGMOIDACTIVATION_H_

#include "ActivationFunction.h"
#include <cmath>

namespace Network {
class SigmoidActivation : public ActivationFunction {
  [[nodiscard]] double compute(double x) const override {
    return 1.0 / (1.0 + std::exp(-x));
  }
  [[nodiscard]] double computeDerivative(double x) const override {
    double sigmoid = compute(x);
    return sigmoid * (1 - sigmoid);
  }
};
}// namespace Network

#endif//SIGMOIDACTIVATION_H_
