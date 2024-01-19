#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_
#include "Eigen/Dense"
#include <cmath>
#include <vector>

namespace Network {
using Vector = Eigen::VectorXd;

class LossFunction {
 public:
  virtual ~LossFunction() = default;
  virtual double computeLoss(const Vector &predicted, const Vector &actual) const = 0;
  virtual Vector computeDerivativeLoss(const Vector &predicted, const Vector &actual) const = 0;
};
}// namespace Network

#endif//LOSSFUNCTION_H_
