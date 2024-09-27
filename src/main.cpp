#include "CrossEntropyLoss.h"
#include "MNISTLoader.h"
#include "MeanSquaredError.h"
#include "Network.h"
#include "ReluActivation.h"
#include "SigmoidActivation.h"
#include "SoftmaxActivation.h"
#include "TanhActivation.h"
#include <iostream>
#include <vector>

int main() {
  // Пути к файлам MNIST
  std::string trainImagesPath = "../data/train-images.idx3-ubyte";
  std::string trainLabelsPath = "../data/train-labels.idx1-ubyte";
  std::string testImagesPath = "../data/t10k-images.idx3-ubyte";
  std::string testLabelsPath = "../data/t10k-labels.idx1-ubyte";
  // Загрузка обучающих данных
  Data::MNISTLoader trainLoader(trainImagesPath, trainLabelsPath);
  if (!trainLoader.loadData()) {
    std::cerr << "Ошибка при загрузке обучающих данных MNIST." << std::endl;
    return 1;
  }
  // Загрузка тестовых данных
  Data::MNISTLoader testLoader(testImagesPath, testLabelsPath);
  if (!testLoader.loadData()) {
    std::cerr << "Ошибка при загрузке тестовых данных MNIST." << std::endl;
    return 1;
  }
  // Получение обучающего датасета
  auto trainDataset = trainLoader.getDataset();
  std::vector<Eigen::VectorXd> trainInputs;
  std::vector<Eigen::VectorXd> trainTargets;
  for (const auto &dataPair : trainDataset) {
    trainInputs.push_back(dataPair.first);
    trainTargets.push_back(dataPair.second);
  }
  // Получение тестового датасета
  auto testDataset = testLoader.getDataset();
  std::vector<Eigen::VectorXd> testInputs;
  std::vector<Eigen::VectorXd> testTargets;
  for (const auto &dataPair : testDataset) {
    testInputs.push_back(dataPair.first);
    testTargets.push_back(dataPair.second);
  }
  // Создание функции активации и функции потерь
  auto sigmoidActivation = Network::SigmoidActivation();
  auto softmaxActivation = Network::SoftmaxActivation();
  auto tanhActivation = Network::TanhActivation();
  auto reluActivation = Network::ReLUActivation();

  auto MeanSquaredError = Network::MeanSquaredError();
  auto CrossEntropy = Network::CrossEntropyLoss();

  // Создание нейросети
  Network::NeuralNetwork neuralNetwork(&CrossEntropy);

  // Добавление слоев
  neuralNetwork.addLayer(Network::Layer(784, 128, reluActivation));
  neuralNetwork.addLayer(Network::Layer(128, 64, reluActivation));
  neuralNetwork.addLayer(Network::Layer(64, 10, softmaxActivation));

  // Параметры обучения
  int epochs = 5;
  double learningRate = 0.5;
  int batchSize = 32;

  // Обучение нейросети
  neuralNetwork.train(trainInputs, trainTargets, epochs, learningRate, batchSize);

  // Тестирование нейросети
  int correct = 0;
  int total = testInputs.size();
  for (size_t i = 0; i < testInputs.size(); ++i) {
    Eigen::VectorXd output = neuralNetwork.predict(testInputs[i]);
    int predictedLabel;
    output.maxCoeff(&predictedLabel);
    int actualLabel;
    testTargets[i].maxCoeff(&actualLabel);
    if (predictedLabel == actualLabel) {
      ++correct;
    }
  }
  double accuracy = static_cast<double>(correct) / total * 100.0;
  std::cout << "Точность на тестовом наборе: " << accuracy << "%" << std::endl;

  return 0;
}
