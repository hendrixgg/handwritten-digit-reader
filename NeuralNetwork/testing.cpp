#include "NeuralNet.h"
#include <vector>
#include <cstdio>
#include <chrono>

struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

// NeuralNet digitReader({10, 16, 16, 784});
// NeuralNet digitReader("savedNeuralNetwork.bin");
NeuralNet digitReader("currentNeuralNetwork.bin");

TestData data;

int main() {
    // try out on test data
    FILE* testDatafile = fopen("../TestData/TestData.bin", "rb");
    fread(&data, sizeof(TestData), 1, testDatafile);
    fclose(testDatafile);
    
    double avgCost = 0;
    const int numTests = data.size, startPos = 0;
    printf("[Program Start]\n");
    auto begin = std::chrono::steady_clock::now();
    
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
        double maxVal = -1e9;
        int answer = 0;
        for(int i = 0; i < output.size(); ++i) {
            if(output[i] > maxVal) {
                maxVal = output[i], answer = i;
            }
        }
        std::vector<double> expected(10);
        expected[data.labels[t]] = 1;
        avgCost += digitReader.error(expected);
        correctAnswers += answer==data.labels[t];
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
    
    // digitReader.saveToFile("savedNeuralNetwork.bin");
}