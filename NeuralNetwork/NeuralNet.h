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
    // 0-th layer is the output layer, 1-th layer is the last hidden layer, ...
    int inputSize, numberOfLayers;
    std::vector<int> nodesInLayer;
    std::vector<std::vector<double>> z_value;
    std::vector<std::vector<double>> value; // value of a node v[L][i] = value of the i-th node on the L-th layer
    // weights matrix, W[L][j][k] = weight from (k-th node) of (layer L+1) to (j-th node) of (layer L)
    std::vector<std::vector<std::vector<double>>> weight;
    // bias matrix, b[L][k] = bias on k-th node on L-th layer
    std::vector<std::vector<double>> bias;

    // constructs a neural net with the specified structure containing random weights and biases
    NeuralNet(const int inpSize, const int numOfLayers, const std::vector<int>& dimensions): inputSize(inpSize), numberOfLayers(numOfLayers), nodesInLayer(dimensions) {
        // put in a new weight matrix for each layer in the network
        for(int l = 0; l + 1 < numberOfLayers; ++l) {
            value.emplace_back(nodesInLayer[l]);
            z_value.emplace_back(nodesInLayer[l]);
            weight.emplace_back(nodesInLayer[l], std::vector<double>(nodesInLayer[l + 1]));
            bias.emplace_back(nodesInLayer[l]);
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                for(int k = 0; k < nodesInLayer[l + 1]; ++k) {
                    weight[l][j][k] = random(-1, 1); // random value to be put as a weight
                }
                bias[l][j] = random(-1, 1);
            }
        }
        // first hidden layer
        const int nodesInFirstLayer = nodesInLayer[numberOfLayers - 1];
        value.emplace_back(nodesInFirstLayer);
        z_value.emplace_back(nodesInFirstLayer);
        weight.emplace_back(nodesInFirstLayer, std::vector<double>(inputSize));
        bias.emplace_back(nodesInFirstLayer);
        for(int j = 0; j < nodesInFirstLayer; ++j) {
            for(int k = 0; k < inputSize; ++k) {
                weight[numberOfLayers - 1][j][k] = random(-1, 1); // random value to be put as a weight
            }
            bias[numberOfLayers - 1][j] = random(-1, 1);
        }
        value.emplace_back(inputSize); // for input layer
    }

    // constructs a neural net from the source file containing structure, weights and biases
    NeuralNet(const char* sourceFilePath) {
        FILE* sourceFile = fopen(sourceFilePath, "rb");
        
        // read dimesions of neural network
        fread(&inputSize, sizeof(int), 1, sourceFile);
        fread(&numberOfLayers, sizeof(int), 1, sourceFile);
        nodesInLayer.resize(numberOfLayers);
        fread(nodesInLayer.data(), sizeof(int), numberOfLayers, sourceFile);

        // read weights and biases and structure the vectors to store the values
        for(int l = 0; l + 1 < numberOfLayers; ++l) {
            value.emplace_back(nodesInLayer[l]);
            z_value.emplace_back(nodesInLayer[l]);
            weight.emplace_back(nodesInLayer[l], std::vector<double>(nodesInLayer[l + 1]));
            bias.emplace_back(nodesInLayer[l]);
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                fread(&weight[l][j][0], sizeof(double), nodesInLayer[l+1], sourceFile);
                fread(&bias[l][j], sizeof(double), 1, sourceFile);
            }
        }
        // first hidden layer
        const int nodesInFirstHiddenLayer = nodesInLayer[numberOfLayers - 1];
        value.emplace_back(nodesInFirstHiddenLayer);
        z_value.emplace_back(nodesInFirstHiddenLayer);
        weight.emplace_back(nodesInFirstHiddenLayer, std::vector<double>(inputSize));
        bias.emplace_back(nodesInFirstHiddenLayer);
        for(int j = 0; j < nodesInFirstHiddenLayer; ++j) {
            fread(&weight[numberOfLayers - 1][j][0], sizeof(double), inputSize, sourceFile);
            fread(&bias[numberOfLayers - 1][j], sizeof(double), 1, sourceFile);
        }
        value.emplace_back(inputSize); // for input layer

        fclose(sourceFile);
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
    void saveToFile(const char * filePath) {
        FILE* saveFile = fopen(filePath, "wb");
        
        // write dimensions of neural network
        fwrite(&inputSize, sizeof(int), 1, saveFile);
        fwrite(&numberOfLayers, sizeof(int), 1, saveFile);
        fwrite(&nodesInLayer[0], sizeof(int), nodesInLayer.size(), saveFile);

        // layer
        for(int l = 0; l < numberOfLayers; ++l) {
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
    std::vector<double> operator ()(const std::vector<double>& input) {
        if(input.size() != inputSize) {
            printf("ERROR: Input size not valid for neural network. Input an std::vector<double> with size %d. Operation terminated.\n", inputSize);
            return {-999};
        }
        
        value[numberOfLayers].assign(input.begin(), input.end()); // input layer
        int l = numberOfLayers-1, previousLayerSize = inputSize;
        while (l >= 0) {
            for(int j = 0; j < nodesInLayer[l]; ++j) {
                double z = bias[l][j];
                for(int k = 0; k < previousLayerSize; ++k) {
                    z += weight[l][j][k] * value[l+1][k];
                }
                z_value[l][j] = z;
                value[l][j] = z > 0 ? z : 0;
            }
            previousLayerSize = nodesInLayer[l--];
        }

        return value[0];
    }

    // returns the cost of an operation
    double error(const std::vector<double>& expected) {
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
};