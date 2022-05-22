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

    int batchSize = 100;
    for(int gen = 0; gen < 600; ++gen) {
        int startPos = random(0, data.size - batchSize - 1);
        std::vector<std::vector<unsigned char>> trainingBatch(batchSize, std::vector<unsigned char>(784));
        std::vector<std::vector<unsigned char>> expectedOutput(batchSize, std::vector<unsigned char>(10));
        for(int i = startPos; i < startPos + batchSize; ++i) {
            trainingBatch[i].assign(data.images[i], data.images[i]+784);
            expectedOutput[i][data.labels[i]] = 1U;
        }
        trainer.train(trainingBatch, expectedOutput, 0.8);
    }
}