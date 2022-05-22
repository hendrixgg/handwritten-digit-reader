#include "NeuralNet.h"
#include "Trainer.h"

#include <vector>
#include <cstdio>
#include <chrono>
#include <random>

std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
// generates a random integer on the interval [range_from, range_to]
int random(const int& range_from, const int& range_to)
{
    std::uniform_int_distribution<int>    distribution(range_from, range_to);
    return distribution(generator);
}

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
    
    printf("[Program Start]\n");
    auto begin = std::chrono::steady_clock::now();
    
    int batchSize = 100;
    for(int gen = 0; gen < 1; ++gen) {
        int startPos = random(0, data.size - batchSize - 1);
        std::vector<std::vector<double>> trainingBatch(batchSize, std::vector<double>(784));
        std::vector<std::vector<double>> expectedOutput(batchSize, std::vector<double>(10));
        for(int i = 0; i < batchSize; ++i) {
            trainingBatch[i].assign(data.images[i+startPos], data.images[i+startPos]+784);
            expectedOutput[i][data.labels[i+startPos]] = 1U;
        }
        trainer.train(trainingBatch, expectedOutput, 0.8);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    printf("[Time Elasped: %lld ms]\n", duration.count());

    digitReader.saveToFile("currentNeuralNetwork.bin");
    
}