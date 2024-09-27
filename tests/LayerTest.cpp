#include "Layer.h"
#include "ReluActivation.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {

TEST(LayerTest, Initialization) {
  Network::ReLUActivation relu;
  Network::Layer layer(3, 2, relu);
  EXPECT_EQ(layer.getWeights().rows(), 2);
  EXPECT_EQ(layer.getWeights().cols(), 3);
  EXPECT_EQ(layer.getBiases().size(), 2);
  EXPECT_TRUE(layer.getWeights().allFinite());
  EXPECT_TRUE(layer.getBiases().allFinite());
}

TEST(LayerTest, ForwardPass) {
  Network::ReLUActivation relu;
  Network::Layer layer(3, 2, relu);
  Eigen::MatrixXd input(3, 1);
  input << 1.0, -1.0, 2.0;
  Eigen::MatrixXd output = layer.forward(input);
  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 1);
  EXPECT_TRUE(output.allFinite());
}

TEST(LayerTest, BackwardPass) {
  Network::ReLUActivation relu;
  Network::Layer layer(3, 2, relu);
  Eigen::MatrixXd input(3, 1);
  input << 1.0, -1.0, 2.0;
  layer.forward(input);
  Eigen::MatrixXd gradOutput(2, 1);
  gradOutput << 0.1, -0.2;
  Eigen::MatrixXd gradInput = layer.backward(gradOutput);
  EXPECT_EQ(gradInput.rows(), 3);
  EXPECT_EQ(gradInput.cols(), 1);
  EXPECT_TRUE(gradInput.allFinite());
  EXPECT_TRUE(layer.getWeights().allFinite());
  EXPECT_TRUE(layer.getBiases().allFinite());
}

TEST(LayerTest, UpdateWeights) {
  Network::ReLUActivation relu;
  Network::Layer layer(3, 2, relu);
  Eigen::MatrixXd input(3, 1);
  input << 1.0, -1.0, 2.0;
  layer.forward(input);
  Eigen::MatrixXd gradOutput(2, 1);
  gradOutput << 0.1, -0.2;
  layer.backward(gradOutput);
  double learningRate = 0.01;
  int batchSize = 1;
  layer.updateWeights(learningRate, batchSize);
  EXPECT_TRUE(layer.getWeights().allFinite());
  EXPECT_TRUE(layer.getBiases().allFinite());
}

}// namespace
