#ifndef TRAINERB_TRAINER_H
#define TRAINERB_TRAINER_H

#include "NeuralNet.h"

class Trainer {
    NeuralNet* net;

    std::vector<std::vector<std::vector<double>>> weightGradient;
    std::vector<std::vector<double>> biasGradient;

    inline double partialA_wto_Z(int l, int j) { return net->value[l][j] * (1 - net->value[l][j]);}

public:
    Trainer(NeuralNet* nn);

    void train(const std::vector<std::vector<double>>& trainingExamples, const std::vector<std::vector<double>>& expected, const double rate);

    void backProp(const std::vector<double>& example, const std::vector<double>& expected);
};

#endif // TRAINERB_TRAINER_H