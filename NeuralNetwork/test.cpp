#include "NeuralNet.h"
#include <vector>
#include <cstdio>
#include <chrono>
#include <algorithm>

struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

NeuralNet digitReader({10, 16, 16, 784});
TestData data;

int main() {
    // try out on test data
    FILE* testDatafile = fopen("../TestData/TestData.bin", "rb");
    fread(&data, sizeof(TestData), 1, testDatafile);
    fclose(testDatafile);

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

    printf("[Program Start]\n");
    auto begin = std::chrono::steady_clock::now();
    
    double avgCost = 0;
    const int numTests = data.size, startPos = 0;
    int correctAnswers = 0;

    for(int t = startPos; t < startPos + numTests; ++t) {
        // print image
        // printf("label: %d\n", data.labels[t]);
        // for(int i = 0; i < 28; ++i) {
        //     for(int j = 0; j < 28; ++j) {
        //         printf("%4d", data.images[t][i*28 + j]);
        //     }
        //     printf("\n");
        // }

        // find the network's answer
        std::vector<double> output(digitReader(std::vector<double>(data.images[t], data.images[t]+784)));
        int answer = std::max_element(output.begin(), output.end()) - output.begin();
        std::vector<double> expected(10);
        expected[data.labels[t]] = 1;
        avgCost += digitReader.error(expected);
        correctAnswers += answer==int(data.labels[t]);

        // display results
        // printf("label: %d\n", data.labels[t]);
        // printf("network output:\n");
        // for(int i = 0; i < output.size(); ++i) {
        //     printf("%d: %.2lf %s\n", i, output[i], (i == answer ? "<--" : ""));
        // }
        // printf("cost: %lf\n", digitReader.error(expected));
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    printf("[Time Elasped: %lld ms]\n", duration.count());

    printf("average cost: %lf\n", avgCost/numTests);
    printf("%c correct: %lf\n", '%', correctAnswers * 1.0 / numTests);
    
    printf("save neural network? (y/n): ");
    char wantToSave[10];
    scanf("%s", wantToSave);
    if(wantToSave[0] == 'y' || wantToSave[0] == 'Y')
        digitReader.saveToFile("savedNeuralNetwork.bin");
}