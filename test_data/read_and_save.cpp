#include <stdio.h>

int ChangeEndianness(int value) {
    int result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}
struct TestData {
    int size = 10000;
    unsigned char labels[10000];
    unsigned char images[10000][784];
};

TestData data;

int main() {
    FILE* labels_file = fopen("t10k-labels.idx1-ubyte", "rb");
    FILE* images_file = fopen("t10k-images.idx3-ubyte", "rb");
    FILE* save_file =  fopen("test_data.bin", "wb");

    int tmp[10];
    // read labels
    fread(tmp, 4, 2, labels_file);// magic number and number of items (in wrong endianness, use ChangeEndianness to see proper values)
    fread(data.labels, 1, data.size, labels_file);
    fclose(labels_file);
    // read images
    fread(tmp, 4, 4, images_file);
    fread(data.images, 1, data.size*784, images_file);
    fclose(images_file);
    // write to save_file
    fwrite(&data, sizeof(TestData), 1, save_file);
    fclose(save_file);
}