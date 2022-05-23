#include "NeuralNet.h"
#include "Trainer.h"

#include <vector>
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>

struct TrainData {
    int size = 60000;
    unsigned char labels[60000];
    unsigned char images[60000][784];
};

NeuralNet digitReader({10, 16, 16, 784});
Trainer trainer(&digitReader);
TrainData data;

int main() {
    FILE* trainDatafile = fopen("../TrainData/TrainData.bin", "rb");
    fread(&data, sizeof(TrainData), 1, trainDatafile);
    fclose(trainDatafile);

    int whichNet = 0;
    puts("[0] - for new neural net");
    puts("[1] - to load from savedNeuralNetwork.bin");
    puts("[2] - to load from currentNeuralNetwork.bin");
    scanf("%d", &whichNet);
    switch (whichNet)
    {
    case 1:
        digitReader.initFromFile("savedNeuralNetwork.bin");
        break;
    case 2:
        digitReader.initFromFile("currentNeuralNetwork.bin");
    default:
        break;
    }
    
    int numberOfBatches;
    printf("Enter number of training batches to run: ");
    scanf("%d", &numberOfBatches);

    printf("\n[Program Start]\n");
    auto begin = std::chrono::steady_clock::now();
    
    int batchSize = 100;
    std::vector<int> shuffle(data.size);
    std::iota(shuffle.begin(), shuffle.end(), 0);
    std::random_shuffle(shuffle.begin(), shuffle.end());
    
    for(int batch = 0, t = 0; batch < numberOfBatches; ++batch, t+=batchSize) {
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