#include "SoftmaxActivation.h"

namespace Network {
Matrix SoftmaxActivation::compute(const Matrix &matrix) const {
  Matrix tmp = (matrix.rowwise() - matrix.colwise().maxCoeff()).array().exp().matrix();
  return ((tmp.array() / ((Vector::Ones(matrix.rows()) * tmp.colwise().sum()).array()))).matrix();
}

Matrix SoftmaxActivation::computeDerivative(const Matrix &matrix) const {
  Matrix softmaxMatrix = compute(matrix);
  Matrix grad(matrix.rows(), matrix.cols());
  for (int col = 0; col < matrix.cols(); ++col) {
    Vector s = softmaxMatrix.col(col);
    Matrix diag_s = s.asDiagonal();
    Matrix jacobian = diag_s - (s * s.transpose());
    grad.col(col) = jacobian.diagonal();
  }
  return grad;
}

}// namespace Network
