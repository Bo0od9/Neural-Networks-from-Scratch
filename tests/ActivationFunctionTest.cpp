#include "ReluActivation.h"
#include "SigmoidActivation.h"
#include "SoftmaxActivation.h"
#include "TanhActivation.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {

TEST(SigmoidActivationTest, Compute) {
  Network::SigmoidActivation sigmoid;
  Eigen::VectorXd input(3);
  input << -1.0, 0.0, 1.0;
  Eigen::VectorXd expected_output(3);
  expected_output << 1.0 / (1.0 + std::exp(1.0)), 0.5, 1.0 / (1.0 + std::exp(-1.0));
  Eigen::VectorXd output = sigmoid.compute(input);
  EXPECT_TRUE(output.isApprox(expected_output, 1e-6));
}

TEST(SigmoidActivationTest, ComputeDerivative) {
  Network::SigmoidActivation sigmoid;
  Eigen::VectorXd input(3);
  input << -2.0, 0.0, 2.0;
  Eigen::VectorXd sigmoid_output = sigmoid.compute(input);
  Eigen::VectorXd expected_derivative = sigmoid_output.array() * (1 - sigmoid_output.array());
  Eigen::VectorXd derivative = sigmoid.computeDerivative(input);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

TEST(ReLUActivationTest, Compute) {
  Network::ReLUActivation relu;
  Eigen::VectorXd input(5);
  input << -2.0, -0.5, 0.0, 0.5, 2.0;
  Eigen::VectorXd expected_output(5);
  expected_output << 0.0, 0.0, 0.0, 0.5, 2.0;
  Eigen::VectorXd output = relu.compute(input);
  EXPECT_TRUE(output.isApprox(expected_output, 1e-6));
}

TEST(ReLUActivationTest, ComputeDerivative) {
  Network::ReLUActivation relu;
  Eigen::VectorXd input(5);
  input << -2.0, -0.5, 0.0, 0.5, 2.0;
  Eigen::VectorXd expected_derivative(5);
  expected_derivative << 0.0, 0.0, 0.0, 1.0, 1.0;
  Eigen::VectorXd derivative = relu.computeDerivative(input);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

TEST(TanhActivationTest, Compute) {
  Network::TanhActivation tanh_act;
  Eigen::VectorXd input(3);
  input << -1.0, 0.0, 1.0;
  Eigen::VectorXd expected_output = input.array().tanh();
  Eigen::VectorXd output = tanh_act.compute(input);
  EXPECT_TRUE(output.isApprox(expected_output, 1e-6));
}

TEST(TanhActivationTest, ComputeDerivative) {
  Network::TanhActivation tanh_act;
  Eigen::VectorXd input(3);
  input << -1.0, 0.0, 1.0;
  Eigen::VectorXd tanh_output = tanh_act.compute(input);
  Eigen::VectorXd expected_derivative = 1.0 - tanh_output.array().square();
  Eigen::VectorXd derivative = tanh_act.computeDerivative(input);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

TEST(SoftmaxActivationTest, Compute) {
  Network::SoftmaxActivation softmax;
  Eigen::MatrixXd input(3, 2);
  input << 1.0, 2.0, 2.0, 1.0, 3.0, 0.0;
  Eigen::MatrixXd output = softmax.compute(input);
  Eigen::RowVectorXd col_sums = output.colwise().sum();
  Eigen::RowVectorXd expected_sums = Eigen::RowVectorXd::Ones(output.cols());
  EXPECT_TRUE(col_sums.isApprox(expected_sums, 1e-6));
}

TEST(SoftmaxActivationTest, ComputeDerivative) {
  Network::SoftmaxActivation softmax;
  Eigen::MatrixXd input(3, 1);
  input << 1.0, 2.0, 3.0;
  Eigen::MatrixXd derivative = softmax.computeDerivative(input);
  EXPECT_EQ(derivative.rows(), 3);
  EXPECT_EQ(derivative.cols(), 1);
}

}// namespace
