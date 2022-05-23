#include "Trainer.h"

#include <stdio.h>

inline double Trainer::partialA_wrt_Z(int l, int j) { return net->value[l][j] * (1 - net->value[l][j]);}

Trainer::Trainer(NeuralNet* nn):  net(nn) {
    if(net == NULL) {
        printf("ERROR: NeuralNet* is NULL");
        return;
    }
    
    for(int l = 0; l + 1 < net->numberOfLayers; ++l) {
        weightGradient.emplace_back(net->nodesInLayer[l], std::vector<double>(net->nodesInLayer[l + 1]));
        biasGradient.emplace_back(net->nodesInLayer[l]);
        valueGradient.emplace_back(net->nodesInLayer[l]);
    }
}

void Trainer::train(const std::vector<std::vector<double>>& trainingExamples, const std::vector<std::vector<double>>& expected, const double rate) {
    // clear gradient matrixes
    for(int l = 0; l + 1 < net->numberOfLayers; ++l) {
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            biasGradient[l][j] = valueGradient[l][j] = 0;
            for(int k = 0; k < net->nodesInLayer[l + 1]; ++k) 
                weightGradient[l][j][k] = 0;
        }
    }

    // find gradient
    for(size_t i = 0; i < trainingExamples.size(); ++i) {
        backProp(trainingExamples[i], expected[i]);
    }
    
    // modify weights and biases based on gradients
    for(int l = 0; l + 1 < net->numberOfLayers; ++l) {
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            net->bias[l][j] -= rate * biasGradient[l][j] / trainingExamples.size(); // average gradient
            for(int k = 0; k < net->nodesInLayer[l + 1]; ++k) {
                net->weight[l][j][k] -= rate * weightGradient[l][j][k] / trainingExamples.size();
            }
        }
    }
}

void Trainer::backProp(const std::vector<double>& example, const std::vector<double>& expected) {
    // run operation to get values
    (*net)(example);
    
    // calculate value gradient
    for(int j = 0; j < net->nodesInLayer[0]; ++j) {
        valueGradient[0][j] = (net->value[0][j]) - expected[j];
    }
    for(int l = 1; l < net->numberOfLayers - 1; ++l) {
        for(int k = 0; k < net->nodesInLayer[l]; ++k) {
            valueGradient[l][k] = 0;
            for(int j = 0; j < net->nodesInLayer[l-1]; ++j) {
                valueGradient[l][k] += (net->weight[l-1][j][k]) * partialA_wrt_Z(l-1, j) * valueGradient[l-1][j];
            }
        }
    }

    // calculate bias gradient
    for (int l = 0; l < net->numberOfLayers - 1; ++l) {
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            biasGradient[l][j] += partialA_wrt_Z(l, j) * valueGradient[l][j];
        }
    }

    // caluculate weight gradient
    for(int l = 0; l + 1 < net->numberOfLayers; ++l) {
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            for(int k = 0; k < net->nodesInLayer[l + 1]; ++k) {
                weightGradient[l][j][k] += net->value[l][k] * partialA_wrt_Z(l, j) * valueGradient[l][j];
            }
        }
    }
}