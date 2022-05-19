#include "NeuralNet.h"

struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

NeuralNet digitReader(784, 3, {16, 16, 10});
TestData data;

int main() {
    // try out on test data
    FILE* testDatafile = fopen("../test_data/test_data.bin", "rb");
    fread(&data, sizeof(TestData), 1, testDatafile);
    
    // run test case through network
    std::vector<double> output(digitReader(std::vector<double>(data.images[0], data.images[0]+784)));
    
    // find answer
    int maxVal = -1e9, answer = 0;
    for(int i = 0; i < output.size(); ++i) {
        if(output[i] > maxVal) {
            maxVal = output[i], answer = i;
        }
    }

    // display results
    printf("label: %d\n", data.labels[0]);
    printf("network output:\n");
    for(int i = 0; i < output.size(); ++i) {
        printf("%d: %.2lf %s\n", i, output[i], (i == answer ? "<--" : ""));
    }
    fclose(testDatafile);

    // save the neural network
    char wantToSave;
    printf("do you want to save the current neural network? This will overwrite the existing save file in this directory.\n");
    printf("(Y/N): ");
    scanf("%c", &wantToSave);
    if(wantToSave == 'y' || wantToSave == 'Y') {
        FILE* savedNeuralNet = fopen("savedNeuralNetwork.bin", "wb");
        fwrite(&digitReader, sizeof(digitReader), 1, savedNeuralNet);
        fclose(savedNeuralNet);
    }
}