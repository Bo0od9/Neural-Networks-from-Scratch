# Курсовой проект «Нейросети с нуля»
Студент: Плешков Иван Игоревич

Группа: БКНАД221

# Сборка
* Установите необходимые библиотеки:
```
git submodule update --init --recursive
```
* Для сборки и запуска проекта в корневой директории выполните следующие команды:
```shell
mkdir build
cd build
cmake ..
make
./src/NeuralNetwork
```
* Для запуска тестов выполните следующие команды:
```shell
mkdir buildtest
cd buildtest
cmake -DBUILD_TESTS=ON -DBUILD_MAIN_APP=OFF ..
make
ctest
```
В файле ```main.cpp``` есть пример использования библиотеки.