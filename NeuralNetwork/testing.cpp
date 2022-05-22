#include "NeuralNet.h"
#include <vector>
#include <cstdio>

struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

// NeuralNet digitReader({10, 16, 16, 784});
NeuralNet digitReader("savedNeuralNetwork.bin");
// Trainer trainer(&digitReader);
TestData data;

int main() {
    // try out on test data
    FILE* testDatafile = fopen("../TestData/TestData.bin", "rb");
    fread(&data, sizeof(TestData), 1, testDatafile);
    fclose(testDatafile);
    
    char userInput;
    while (true){
        double avgCost = 0;
        int numTests = data.size;
        for(int t = 0; t < data.size; ++t) {
            std::vector<double> input(data.images[t], data.images[t]+784);
            // for(double& d : input) d /= 255; // make pixel values from 0 to 1 not 0 to 255
            // run test case through network
            std::vector<double> output(digitReader(input));

            // find answer
            double maxVal = -1e9;
            int answer = 0;
            for(int i = 0; i < output.size(); ++i) {
                if(output[i] > maxVal) {
                    maxVal = output[i], answer = i;
                }
            }
            std::vector<double> expected(10);
            expected[data.labels[0]] = 1;
            avgCost += digitReader.error(expected);

            // // display results
            // printf("label: %d\n", data.labels[t]);
            // printf("network output:\n");
            // for(int i = 0; i < output.size(); ++i) {
            //     printf("%d: %.2lf %s\n", i, output[i], (i == answer ? "<--" : ""));
            // }
            // printf("cost: %lf\n", digitReader.error(expected));
        }
        printf("average cost: %lf\n", avgCost/numTests);

        // save the neural network
        printf("do you want to save the current neural network? This will overwrite the existing save file in this directory.\n");
        printf("(Y/N, q/Q to quit): ");
        scanf("%c", &userInput);
        fflush(stdin);
        
        if(userInput == 'q' || userInput == 'Q') 
            break;
        if(userInput == 'y' || userInput == 'Y') {
            digitReader.saveToFile("savedNeuralNetwork.bin");
        }
        digitReader.initRandom();
    }
    
}