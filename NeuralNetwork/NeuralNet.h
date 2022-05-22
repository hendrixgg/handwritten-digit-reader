#ifndef NEURALNETB_NEURALNET_H
#define NEURALNETB_NEURALNET_H

#include <vector>

class NeuralNet {
    // generates a random real number on the interval [range_from, range_to)
    double static random(const double& range_from, const double& range_to);
    double static f(double z);
public:
    
    // 0-th layer is the output layer, 1-th layer is the last hidden layer, ...
    int numberOfLayers; // input layer + hidden layers + output layer
    std::vector<int> nodesInLayer;
    std::vector<std::vector<double>> value; // value of a node v[L][i] = value of the i-th node on the L-th layer
    // weights matrix, W[L][j][k] = weight from (k-th node) of (layer L+1) to (j-th node) of (layer L)
    std::vector<std::vector<std::vector<double>>> weight;
    // bias matrix, b[L][k] = bias on k-th node on L-th layer
    std::vector<std::vector<double>> bias;

    // constructs a neural net with the specified structure containing random weights and biases
    NeuralNet(const std::vector<int>& dimensions);

    // constructs a neural net from the source file created by saveToFile function
    NeuralNet(const char* sourceFilePath);

    // initializes the weights and biases to random values
    void initRandom(const std::vector<int>& dimensions);
    void initRandom();

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
    void saveToFile(const char* filePath);

    // given an input vector, returns the values in the last layer of the network
    std::vector<double> operator ()(const std::vector<double>& input);

    // returns the cost of an operation
    double error(const std::vector<double>& expected);
};

#endif //NEURALNETB_NEURALNET_H