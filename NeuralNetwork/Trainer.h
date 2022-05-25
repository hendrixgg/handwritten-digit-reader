#pragma once

#include "NeuralNet.h"
#include <vector>

class Trainer {
    NeuralNet* net;

    std::vector<std::vector<std::vector<double>>> weightGradient;
    std::vector<std::vector<double>> biasGradient;
    std::vector<std::vector<double>> valueGradient;

public:
    Trainer(NeuralNet* nn);

    void train(const std::vector<std::vector<double>>& trainingExamples, const std::vector<std::vector<double>>& expected, const double rate);
    
private:
    void backProp(const std::vector<double>& example, const std::vector<double>& expected);
};