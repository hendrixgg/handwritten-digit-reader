#include "NeuralNet.h"
#include "Trainer.h"

#include <vector>
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>

struct TrainData{
    int size = 60000;
    unsigned char labels[60000];
    unsigned char images[60000][784];
};

// NeuralNet digitReader({10, 16, 16, 784});
NeuralNet digitReader("currentNeuralNetwork.bin");
Trainer trainer(&digitReader);
TrainData data;
int main() {
    FILE* trainDatafile = fopen("../TrainData/TrainData.bin", "rb");
    fread(&data, sizeof(TrainData), 1, trainDatafile);
    fclose(trainDatafile);
    
    int numberOfGenerations;
    scanf("%d", &numberOfGenerations);

    printf("[Program Start]\n");
    auto begin = std::chrono::steady_clock::now();
    
    int batchSize = 100;
    int shuffle[data.size];
    std::iota(shuffle, shuffle+data.size, 0);
    std::random_shuffle(shuffle, shuffle+data.size);
    for(int gen = 0, t = 0; gen < numberOfGenerations; ++gen, t+=batchSize) {
        std::vector<std::vector<double>> trainingBatch(batchSize, std::vector<double>(784));
        std::vector<std::vector<double>> expectedOutput(batchSize, std::vector<double>(10));
        for(int i = t; i < t + batchSize; ++i) {
            trainingBatch[i - t].assign(data.images[shuffle[i]], data.images[shuffle[i]]+784);
            expectedOutput[i - t][data.labels[shuffle[i]]] = 1.0;
        }
        trainer.train(trainingBatch, expectedOutput, 0.8);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    printf("[Time Elasped: %lld ms]\n", duration.count());

    digitReader.saveToFile("currentNeuralNetwork.bin");
}