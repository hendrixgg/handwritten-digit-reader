#include "NeuralNet.h"
#include "Trainer.h"

#include <vector>
#include <cstdio>
#include <chrono>
#include <numeric>
#include <algorithm>

struct TrainData {
    int size = 60000;
    unsigned char labels[60000];
    unsigned char images[60000][784];
};
struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

NeuralNet digitReader({10, 16, 16, 784});
Trainer trainer(&digitReader);
TrainData trainData;
TestData testData;

/*
Input required:
[1] - for new neural net
[2] - to load from savedNeuralNetwork.bin
[3] - to load from currentNeuralNetwork.bin
[int]

# of training rounds: [int]

# of batches per round: [int]
*/

int main() {
    FILE* trainDatafile = fopen("../TrainData/TrainData.bin", "rb");
    fread(&trainData, sizeof(TrainData), 1, trainDatafile);
    fclose(trainDatafile);

    FILE* testDatafile = fopen("../TestData/TestData.bin", "rb");
    fread(&testData, sizeof(TestData), 1, testDatafile);
    fclose(testDatafile);

    int whichNet = 0;
    puts("[1] - for new neural net");
    puts("[2] - to load from savedNeuralNetwork.bin");
    puts("[3] - to load from currentNeuralNetwork.bin");
    scanf("%d", &whichNet);
    switch (whichNet)
    {
    case 2:
        digitReader.initFromFile("savedNeuralNetwork.bin");
        break;
    case 3:
        digitReader.initFromFile("currentNeuralNetwork.bin");
    default:
        break;
    }
    
    int trainingRounds;
    int batchSize = 100;
    printf("Enter number of training rounds to run: ");
    scanf("%d", &trainingRounds);
    int numberOfBatches;
    printf("\nEnter number of batches per round (max %d): ", trainData.size / batchSize);
    scanf("%d", &numberOfBatches);

    numberOfBatches = std::min(numberOfBatches, trainData.size / batchSize);

    for(int round = 0; round < trainingRounds; ++round) {
        printf("\n[Round %d Start]\n", round+1);
        auto begin = std::chrono::steady_clock::now();
        
        std::vector<int> shuffle(trainData.size);
        std::iota(shuffle.begin(), shuffle.end(), 0);
        std::random_shuffle(shuffle.begin(), shuffle.end());

        // run gradient descent algorithm
        for(int batch = 0, t = 0; batch < numberOfBatches; ++batch, t+=batchSize) {
            std::vector<std::vector<double>> trainingBatch(batchSize, std::vector<double>(784));
            std::vector<std::vector<double>> expectedOutput(batchSize, std::vector<double>(10));
            for(int i = t; i < t + batchSize; ++i) {
                trainingBatch[i - t].assign(trainData.images[shuffle[i]], trainData.images[shuffle[i]]+784);
                expectedOutput[i - t][trainData.labels[shuffle[i]]] = 1.0;
            }
            trainer.train(trainingBatch, expectedOutput, 0.8);
        }

        // test progress
        double totalCost = 0;
        int correctAnswers = 0;

        for(int t = 0; t < testData.size; ++t) {
            std::vector<double> output(digitReader(std::vector<double>(testData.images[t], testData.images[t]+784)));
            int answer = std::max_element(output.begin(), output.end()) - output.begin();
            std::vector<double> expected(10);
            expected[testData.labels[t]] = 1;
            totalCost += digitReader.error(expected);
            correctAnswers += answer==int(testData.labels[t]);
        }

        printf("average cost: %lf\n", totalCost / testData.size);
        printf("%c correct: %lf\n", '%', 100.0 * correctAnswers / testData.size);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        printf("[Time Elasped: %.3lf sec]\n", duration.count()/1000.0);

        digitReader.saveToFile("currentNeuralNetwork.bin");
    }

    puts("\nTraining complete!");
}