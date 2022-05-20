#include <chrono>
#include <random>
#include <vector>
#include <stdio.h>

std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
// generates a random real number on the interval [range_from, range_to)
double random(const double& range_from, const double& range_to)
{
    std::uniform_real_distribution<double> distribution (range_from, range_to);
    return distribution(generator);
}

struct NeuralNet {
    // -1-th layer is considered the input layer
    int inputSize, numberOfLayers;
    std::vector<int> nodesInLayer;
    std::vector<std::vector<double>> value; // value of a node v[L][i] = value of the i-th node on the L-th layer
    // weights matrix, W[L][j][k] = weight from (k-th node) of (layer L-1) to (j-th node) of (layer L)
    std::vector<std::vector<std::vector<double>>> weight;
    // bias matrix, b[L][k] = bias on k-th node on L-th layer
    std::vector<std::vector<double>> bias;

    NeuralNet(int inputSize, int numberOfLayers, std::vector<int> nodesInLayer) {
        this->inputSize = inputSize, this->numberOfLayers = numberOfLayers, this->nodesInLayer.assign(nodesInLayer.begin(), nodesInLayer.end());
        // put in a new weight matrix for each layer in the network
        int l = 0, previousLayerSize = inputSize;
        while(l < numberOfLayers) {
            value.emplace_back(nodesInLayer[l]);
            weight.emplace_back(nodesInLayer[l], std::vector<double>(previousLayerSize));
            bias.emplace_back(nodesInLayer[l]);
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                for(int k = 0; k < previousLayerSize; ++k) {
                    weight[l][j][k] = random(-1, 1); // random value to be put as a weight
                }
                bias[l][j] = random(-1, 1);
            }
            previousLayerSize = nodesInLayer[l++];
        }
    }

    NeuralNet(const char* sourceFilePath) {
        FILE* sourceFile = fopen(sourceFilePath, "rb");
        
        // read dimesions of neural network
        fread(&inputSize, sizeof(int), 1, sourceFile);
        fread(&numberOfLayers, sizeof(int), 1, sourceFile);
        nodesInLayer.resize(numberOfLayers);
        fread(nodesInLayer.data(), sizeof(int), numberOfLayers, sourceFile);

        // read weights and biases
        int l = 0, previousLayerSize = inputSize;
        while(l < numberOfLayers) {
            value.emplace_back(nodesInLayer[l]);
            weight.emplace_back(nodesInLayer[l], std::vector<double>(previousLayerSize));
            bias.emplace_back(nodesInLayer[l]);
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                fread(weight[l][j].data(), sizeof(double), previousLayerSize, sourceFile);
                fread(&bias[l][j], sizeof(double), 1, sourceFile);
            }
            previousLayerSize = nodesInLayer[l++];
        }
    }

    void saveToFile(const char * filePath) {
        FILE* saveFile = fopen(filePath, "wb");
        
        // write dimensions of neural network
        fwrite(&inputSize, sizeof(int), 1, saveFile);
        fwrite(&numberOfLayers, sizeof(int), 1, saveFile);
        fwrite(nodesInLayer.data(), sizeof(int), nodesInLayer.size(), saveFile);

        // layer
        for(int l = 0; l < numberOfLayers; ++l) {
            // node
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                // weights, bias
                fwrite(weight[l][j].data(), sizeof(double), weight[l][j].size(), saveFile);
                fwrite(&bias[l][j], sizeof(double), 1, saveFile); 
            }
        }

        fclose(saveFile);
    }

    std::vector<double> operator ()(std::vector<double> input) {
        if(input.size() != inputSize) {
            printf("Input size not valid for neural network. Input an std::vector<double> with size %d. Operation terminated.\n", inputSize);
            return {-999};
        }
        // input to layer 0
        for(int j = 0; j < value[0].size(); ++j) {
            double z = bias[0][j];
            for(int k = 0; k < inputSize; ++k) {
                z += weight[0][j][k] * input[k];
            }
            value[0][j] = z > 0 ? z : 0;
        }
        // all other layers
        for(int l = 1; l < numberOfLayers; ++l) {
            for(int j = 0; j < value[l].size(); ++j) {
                double z = bias[l][j];
                for(int k = 0; k < value[l-1].size(); ++k) {
                    z += weight[l][j][k] * value[l-1][k];
                }
                value[l][j] = z > 0 ? z : 0;
            }
        }
        return value[numberOfLayers-1];
    }
};