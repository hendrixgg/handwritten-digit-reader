#include <stdio.h>

int ChangeEndianness(int value) {
    int result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}
struct TrainData {
    int size = 60000;
    unsigned char labels[60000];
    unsigned char images[60000][784];
};

TrainData data;

int main() {
    FILE* data_file =  fopen("./TrainData/TrainData.bin", "rb");
    fread(&data, sizeof(TrainData), 1, data_file);
    printf("label: %d\n", data.labels[0]);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            printf("%4d", data.images[0][i*28 + j]);
        }
        printf("\n");
    }
    fclose(data_file);
}