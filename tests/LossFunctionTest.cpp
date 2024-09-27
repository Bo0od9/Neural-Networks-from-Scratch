#include "CrossEntropyLoss.h"
#include "MeanSquaredError.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {

TEST(MeanSquaredErrorTest, ComputeLoss) {
  Network::MeanSquaredError mse;
  Eigen::VectorXd predicted(3);
  predicted << 1.0, 2.0, 3.0;
  Eigen::VectorXd actual(3);
  actual << 1.0, 2.0, 2.0;
  double expected_loss = ((predicted - actual).array().square().sum()) / predicted.size();
  double loss = mse.computeLoss(predicted, actual);
  EXPECT_NEAR(loss, expected_loss, 1e-6);
}

TEST(MeanSquaredErrorTest, ComputeDerivativeLoss) {
  Network::MeanSquaredError mse;
  Eigen::VectorXd predicted(3);
  predicted << 1.0, 2.0, 3.0;
  Eigen::VectorXd actual(3);
  actual << 1.0, 2.0, 2.0;
  Eigen::VectorXd expected_derivative = 2.0 * (predicted - actual) / predicted.size();
  Eigen::VectorXd derivative = mse.computeDerivativeLoss(predicted, actual);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

TEST(CrossEntropyLossTest, ComputeLoss) {
  Network::CrossEntropyLoss cross_entropy;
  Eigen::VectorXd predicted(3);
  predicted << 0.2, 0.5, 0.3;
  Eigen::VectorXd actual(3);
  actual << 0.0, 1.0, 0.0;
  double expected_loss = -(actual.array() * predicted.array().log()).sum();
  double loss = cross_entropy.computeLoss(predicted, actual);
  EXPECT_NEAR(loss, expected_loss, 1e-6);
}

TEST(CrossEntropyLossTest, ComputeDerivativeLoss) {
  Network::CrossEntropyLoss cross_entropy;
  Eigen::VectorXd predicted(3);
  predicted << 0.2, 0.5, 0.3;
  Eigen::VectorXd actual(3);
  actual << 0.0, 1.0, 0.0;
  Eigen::VectorXd expected_derivative = predicted - actual;
  Eigen::VectorXd derivative = cross_entropy.computeDerivativeLoss(predicted, actual);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

TEST(CrossEntropyLossTest, ComputeBatchLoss) {
  Network::CrossEntropyLoss cross_entropy;
  Eigen::MatrixXd predicted(3, 2);
  predicted << 0.2, 0.1, 0.5, 0.7, 0.3, 0.2;
  Eigen::MatrixXd actual(3, 2);
  actual << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
  double expected_loss = -(actual.array() * predicted.array().log()).sum() / predicted.cols();
  double loss = cross_entropy.computeBatchLoss(predicted, actual);
  EXPECT_NEAR(loss, expected_loss, 1e-6);
}

TEST(CrossEntropyLossTest, ComputeBatchDerivativeLoss) {
  Network::CrossEntropyLoss cross_entropy;
  Eigen::MatrixXd predicted(3, 2);
  predicted << 0.2, 0.1, 0.5, 0.7, 0.3, 0.2;
  Eigen::MatrixXd actual(3, 2);
  actual << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
  Eigen::MatrixXd expected_derivative = (predicted - actual) / predicted.cols();
  Eigen::MatrixXd derivative = cross_entropy.computeBatchDerivativeLoss(predicted, actual);
  EXPECT_TRUE(derivative.isApprox(expected_derivative, 1e-6));
}

}// namespace
