#include "Network.h"
#include "MeanSquaredError.h"
#include "ReluActivation.h"
#include "SigmoidActivation.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {

TEST(NeuralNetworkTest, Predict) {
  Network::MeanSquaredError mse;
  Network::NeuralNetwork network(&mse);
  Network::ReLUActivation relu;
  Network::Layer layer1(3, 2, relu);
  network.addLayer(layer1);
  Network::SigmoidActivation sigmoid;
  Network::Layer layer2(2, 1, sigmoid);
  network.addLayer(layer2);
  Eigen::VectorXd input(3);
  input << 1.0, -1.0, 2.0;
  Eigen::VectorXd output = network.predict(input);
  EXPECT_EQ(output.size(), 1);
}

TEST(NeuralNetworkTest, TrainSingleEpoch) {
  Network::MeanSquaredError mse;
  Network::NeuralNetwork network(&mse);
  Network::ReLUActivation relu;
  Network::Layer layer1(3, 2, relu);
  network.addLayer(layer1);
  Network::SigmoidActivation sigmoid;
  Network::Layer layer2(2, 1, sigmoid);
  network.addLayer(layer2);
  std::vector<Eigen::VectorXd> inputs = {
      (Eigen::VectorXd(3) << 1.0, -1.0, 2.0).finished(),
      (Eigen::VectorXd(3) << 2.0, 1.0, -1.0).finished(),
      (Eigen::VectorXd(3) << -1.0, 0.0, 1.0).finished()};
  std::vector<Eigen::VectorXd> targets = {
      (Eigen::VectorXd(1) << 0.5).finished(),
      (Eigen::VectorXd(1) << 1.0).finished(),
      (Eigen::VectorXd(1) << 0.0).finished()};

  ASSERT_NO_THROW(network.train(inputs, targets, 1, 0.01, 2));
}

TEST(NeuralNetworkTest, TrainMultipleEpochs) {
  Network::MeanSquaredError mse;
  Network::NeuralNetwork network(&mse);
  Network::ReLUActivation relu;
  Network::Layer layer1(3, 2, relu);
  network.addLayer(layer1);
  Network::SigmoidActivation sigmoid;
  Network::Layer layer2(2, 1, sigmoid);
  network.addLayer(layer2);
  std::vector<Eigen::VectorXd> inputs = {
      (Eigen::VectorXd(3) << 1.0, -1.0, 2.0).finished(),
      (Eigen::VectorXd(3) << 2.0, 1.0, -1.0).finished(),
      (Eigen::VectorXd(3) << -1.0, 0.0, 1.0).finished()};
  std::vector<Eigen::VectorXd> targets = {
      (Eigen::VectorXd(1) << 0.5).finished(),
      (Eigen::VectorXd(1) << 1.0).finished(),
      (Eigen::VectorXd(1) << 0.0).finished()};

  ASSERT_NO_THROW(network.train(inputs, targets, 5, 0.01, 2));
}

TEST(NeuralNetworkTest, InvalidInput) {
  Network::MeanSquaredError mse;
  Network::NeuralNetwork network(&mse);
  Network::ReLUActivation relu;
  Network::Layer layer1(3, 2, relu);
  network.addLayer(layer1);
  Network::SigmoidActivation sigmoid;
  Network::Layer layer2(2, 1, sigmoid);
  network.addLayer(layer2);
  std::vector<Eigen::VectorXd> invalidInputs = {(Eigen::VectorXd(4) << 1.0, -1.0, 2.0, 3.0).finished()};
  std::vector<Eigen::VectorXd> targets = {(Eigen::VectorXd(1) << 0.5).finished()};

  EXPECT_DEATH(network.train(invalidInputs, targets, 1, 0.01, 1), "Input matrix must have the correct number of rows");
}

}// namespace
