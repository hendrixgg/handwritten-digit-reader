#include "NeuralNet.h"
#include <stdio.h>

class Trainer {
    NeuralNet* net;

    std::vector<std::vector<std::vector<double>>> weightGradient;
    std::vector<std::vector<double>> biasGradient;

public:
    Trainer(NeuralNet* nn):  net(nn) {
        if(net == NULL) {
            printf("ERROR: NeuralNet* is NULL");
            return;
        }
        
        for(int l = 0; l + 1 < net->numberOfLayers; ++l) {
            weightGradient.emplace_back(net->nodesInLayer[l], std::vector<double>(net->nodesInLayer[l + 1]));
            biasGradient.emplace_back(net->nodesInLayer[l]);
        }
    }

    void train(const std::vector<std::vector<double>>& trainingExamples, const std::vector<std::vector<double>>& expected) {

    }

    void backProp(const std::vector<double>& example, const std::vector<double>& expected) {
        // run operation to get values
        (*net)(example);

        // l = 0:
        int l = 0;
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            double tmp_gradient = (net->value[l][j] - expected[j]) * (net->z_value[l][j] > 0);
            biasGradient[l][j] = tmp_gradient;
            for(int k = 0; k < net->nodesInLayer[l + 1]; ++k) {
                weightGradient[l][j][k] = tmp_gradient * (net->value[l + 1][k]);
            }
        }

        // l = 1:
        l = 1;
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            double tmp_gradient = 0;
            for(int m = 0; m < net->nodesInLayer[0]; ++m) {
                tmp_gradient += (net->value[0][m] - expected[m]) * (net->z_value[0][m] > 0) * (net->weight[0][m][j]) * (net->z_value[l][m] > 0);
            }
            biasGradient[l][j] += tmp_gradient;
            for(int k = 0; k < net->nodesInLayer[2]; ++k) {
                weightGradient[l][j][k] += tmp_gradient * (net->value[l + 1][k]);
            }
        }

        // l = 2:
        l = 2;
        for(int j = 0; j < net->nodesInLayer[l]; ++j) {
            // biasGradient[l][j]
            for(int m0 = 0; m0 < net->nodesInLayer[0]; ++m0) {
                double tmp_gradient = 0;
                for(int m1 = 0; m1 < net->nodesInLayer[1]; ++m1) {
                    tmp_gradient += (net->weight[0][m0][m1]) * (net->z_value[l-1][m1] > 0) * (net->weight[l-1][m1][j]) * (net->z_value[l][j] > 0);
                }
                biasGradient[2][j] += (net->value[0][m0] - expected[m0]) * (net->z_value[0][m0] > 0) * tmp_gradient;
            }
            for(int k = 0; k < net->nodesInLayer[l + 1]; ++k) {
                for(int m0 = 0; m0 < net->nodesInLayer[0]; ++m0) {
                    double tmp_gradient = 0;
                    for(int m1 = 0; m1 < net->nodesInLayer[1]; ++m1) {
                        tmp_gradient += (net->weight[0][m0][m1]) * (net->z_value[l-1][m1] > 0) * (net->weight[l-1][m1][j]) * (net->z_value[l][j] > 0) * (net->value[l+1][k]);
                    }
                    weightGradient[2][j][k] += (net->value[0][m0] - expected[m0]) * (net->z_value[0][m0] > 0) * tmp_gradient;
                }
            }
        }
    }
};