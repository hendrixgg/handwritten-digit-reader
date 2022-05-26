#include "NeuralNet.hpp"
#include "Trainer.hpp"

#include <vector>
#include <cstdio>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

struct TrainData
{
    const int size = 60000;
    unsigned char labels[60000];
    unsigned char images[60000][784];
};
struct TestData
{
    const int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

NeuralNet digitReader({10, 150, 300, 784});
Trainer trainer(&digitReader);
TrainData trainData;
TestData testData;

void loadData()
{
    FILE *trainDatafile = fopen("../TrainData/TrainData.bin", "rb");
    fread(&trainData, sizeof(TrainData), 1, trainDatafile);
    fclose(trainDatafile);

    FILE *testDatafile = fopen("../TestData/TestData.bin", "rb");
    fread(&testData, sizeof(TestData), 1, testDatafile);
    fclose(testDatafile);
}

void askWhichNetworkToTrain()
{
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
}

void testProgress()
{
    double totalCost = 0;
    int correctAnswers = 0;

    for (int t = 0; t < testData.size; ++t)
    {
        std::vector<double> output(digitReader(std::vector<double>(testData.images[t], testData.images[t] + 784)));
        int answer = max_element(output.begin(), output.end()) - output.begin();
        std::vector<double> expected(10);
        expected[testData.labels[t]] = 1;
        totalCost += digitReader.error(expected);
        correctAnswers += answer == int(testData.labels[t]);
    }

    printf("average cost: %lf\n", totalCost / testData.size);
    printf("%.2lf %c test error rate\n", 100.0 - 100.0 * correctAnswers / testData.size, '%');
}

void generatePermutation(int *first, int *last)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::iota(first, last, 0);
    std::shuffle(first, last, g);
}

/*
Input required:
[1] - for new neural net
[2] - to load from savedNeuralNetwork.bin
[3] - to load from currentNeuralNetwork.bin
[int]

# of training rounds: [int]

# of batches per round: [int]
*/
int main()
{
    loadData();

    askWhichNetworkToTrain();

    int trainingRounds, numberOfBatches, batchSize = 100;
    // array to contain a permutation of the integers on the interval: [0, trainData.size-1]
    int shuffle[trainData.size];

    printf("Enter number of training rounds to run: ");
    scanf("%d", &trainingRounds);

    printf("\nEnter number of batches per round (max %d): ", trainData.size / batchSize);
    scanf("%d", &numberOfBatches);
    numberOfBatches = std::min(numberOfBatches, trainData.size / batchSize);

    for (int round = 0; round < trainingRounds; ++round)
    {
        printf("\n[Round %d Start]\n", round + 1);
        auto begin = std::chrono::steady_clock::now();

        generatePermutation(shuffle, shuffle + trainData.size);

        // for each batch of training examples
        for (int batch = 0, t = 0; batch < numberOfBatches; ++batch, t += batchSize)
        {
            std::vector<std::vector<double>> trainingExamples(batchSize, std::vector<double>(784));
            std::vector<std::vector<double>> expectedOutput(batchSize, std::vector<double>(10));
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < 784; ++j)
                {
                    trainingExamples[i][j] = trainData.images[shuffle[t + i]][j];
                }
                expectedOutput[i][trainData.labels[shuffle[t + i]]] = 1.0;
            }
            // run gradient descent algorithm
            trainer.train(trainingExamples, expectedOutput, 0.01);
        }

        if ((round + 1) % 10 == 0)
        {
            testProgress();
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        printf("[Time Elasped: %.3lf sec]\n", duration.count() / 1000.0);

        digitReader.saveToFile("currentNeuralNetwork.bin");
    }

    puts("\nTraining complete!");
}