#include "NeuralNet.h"
#include <stdio.h>

class Trainer {
    NeuralNet* net;

    std::vector<std::vector<std::vector<double>>> negWeightGradient;
    std::vector<std::vector<double>> negBiasGradient;

public:
    Trainer(NeuralNet* nn):  net(nn) {
        if(net == NULL) {
            printf("ERROR: NeuralNet* is NULL");
            return;
        }
        int l = 0, previousLayerSize = net->inputSize;
        while(l < net->numberOfLayers) {
            negWeightGradient.emplace_back(net->nodesInLayer[l], std::vector<double>(previousLayerSize));
            negBiasGradient.emplace_back(net->nodesInLayer[l]);
            previousLayerSize = net->nodesInLayer[l++];
        }
    }

    void train(const std::vector<std::vector<double>>& trainingExamples, const std::vector<std::vector<double>>& expected) {

    }

    void backProp(const std::vector<double>& example, const std::vector<double>& expected) {
        
    }
};