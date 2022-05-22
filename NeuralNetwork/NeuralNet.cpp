#include "NeuralNet.h"

#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>

std::mt19937 ____generator__(std::chrono::steady_clock::now().time_since_epoch().count());
// generates a random real number on the interval [range_from, range_to)
double NeuralNet::random(const double& range_from, const double& range_to)
{
    std::uniform_real_distribution<double> distribution (range_from, range_to);
    return distribution(____generator__);
}
double NeuralNet::f(double z) {
    return 1.0 / (1 + std::exp(-z));
}
// constructs a neural net with the specified structure containing random weights and biases
NeuralNet::NeuralNet(const std::vector<int>& dimensions){
    initRandom(dimensions);
}

// constructs a neural net from the source file containing structure, weights and biases
NeuralNet::NeuralNet(const char* sourceFilePath) {
    FILE* sourceFile = fopen(sourceFilePath, "rb");
    
    // read dimesions of neural network
    fread(&numberOfLayers, sizeof(int), 1, sourceFile);
    nodesInLayer.resize(numberOfLayers);
    fread(nodesInLayer.data(), sizeof(int), numberOfLayers, sourceFile);

    // read weights and biases and structure the vectors to store the values
    for(int l = 0; l + 1 < numberOfLayers; ++l) {
        value.emplace_back(nodesInLayer[l]);
        weight.emplace_back(nodesInLayer[l], std::vector<double>(nodesInLayer[l + 1]));
        bias.emplace_back(nodesInLayer[l]);
        for(int j = 0; j < nodesInLayer[l]; ++j) {
            fread(&weight[l][j][0], sizeof(double), nodesInLayer[l+1], sourceFile);
            fread(&bias[l][j], sizeof(double), 1, sourceFile);
        }
    }
    value.emplace_back(nodesInLayer[numberOfLayers-1]); // for input layer

    fclose(sourceFile);
}

void NeuralNet::initRandom(const std::vector<int>& dimensions) {
    numberOfLayers = dimensions.size();
    nodesInLayer.assign(dimensions.begin(), dimensions.end());
    // put in a new weight matrix for each layer in the network
    for(int l = 0; l + 1 < numberOfLayers; ++l) {
        value.emplace_back(nodesInLayer[l]);
        weight.emplace_back(nodesInLayer[l], std::vector<double>(nodesInLayer[l + 1]));
        bias.emplace_back(nodesInLayer[l]);
    }
    value.emplace_back(nodesInLayer[numberOfLayers-1]); // for input layer
    initRandom(); // for random values
}

void NeuralNet::initRandom() {
    // put in a new weight matrix for each layer in the network
    for(int l = 0; l + 1 < numberOfLayers; ++l) {
        for(int j = 0; j < nodesInLayer[l]; ++j) {
            for(int k = 0; k < nodesInLayer[l + 1]; ++k) {
                weight[l][j][k] = random(-1, 1); // random value to be put as a weight
            }
            bias[l][j] = random(-1, 1);
        }
    }
}

/*  saves the neural net in the following format:
    [offset] [type]         [value] [description]
    0000     32 bit integer ??      inputSize
    0004     32 bit integer ??      numberOfLayers (excluding input layer)
    0008     32 bit integer ??      nodesInLayer[0]
    ........
    xxxx     32 bit integer ??      nodesInLayer[numberOfLayers-1]
    xxxx     64 bit double  ??      weight[0][0][0]
    xxxx     64 bit double  ??      weight[0][0][1]
    xxxx     64 bit double  ??      weight[0][0][2]
    ........
    xxxx     64 bit double  ??      weight[layer][node on current layer][node on previous layer]
*/
void NeuralNet::saveToFile(const char * filePath) {
    FILE* saveFile = fopen(filePath, "wb");
    
    // write dimensions of neural network
    fwrite(&numberOfLayers, sizeof(int), 1, saveFile);
    fwrite(&nodesInLayer[0], sizeof(int), nodesInLayer.size(), saveFile);

    // layer
    for(int l = 0; l < numberOfLayers-1; ++l) {
        // node
        for(int j = 0; j < nodesInLayer[l]; ++j) {
            // weights, bias
            fwrite(&weight[l][j][0], sizeof(double), weight[l][j].size(), saveFile);
            fwrite(&bias[l][j], sizeof(double), 1, saveFile); 
        }
    }

    fclose(saveFile);
}

// given an input vector, returns the values in the last layer of the network
template<typename T>
std::vector<double> NeuralNet::operator ()(const std::vector<T>& input) {
    if(input.size() != nodesInLayer[numberOfLayers-1]) {
        printf("ERROR: Input size not valid for neural network. Input an std::vector<double> with size %d. Operation terminated.\n", nodesInLayer[numberOfLayers-1]);
        return {-999};
    }
    
    value[numberOfLayers-1].assign(input.begin(), input.end()); // input layer
    for (int l = numberOfLayers-2; l >= 0; --l) {
        for(int j = 0; j < nodesInLayer[l]; ++j) {
            double z = bias[l][j];
            for(int k = 0; k < nodesInLayer[l+1]; ++k) {
                z += weight[l][j][k] * value[l+1][k];
            }
            value[l][j] = f(z);
        }
    }

    return value[0];
}

// returns the cost of an operation
double NeuralNet::error(const std::vector<double>& expected) {
    if(expected.size() != value[0].size()) {
        printf("ERROR: expected.size() != outputLayerOfNetwork.size(). (%d != %d)\n", expected.size(), value[0].size());
        return -1;
    }
    double err = 0;
    for(int i = 0; i < expected.size(); ++i) {
        double diff = (value[0][i] - expected[i]);
        err += 0.5 * diff * diff;
    }
    return err;
}