#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_
#include "Eigen/Dense"
namespace Network {
class ActivationFunction {

 public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  virtual ~ActivationFunction() = default;
  [[nodiscard]] virtual double compute(double x) const = 0;
  [[nodiscard]] virtual double computeDerivative(double x) const = 0;
};
}// namespace Network
#endif//ACTIVATIONFUNCTION_H_
