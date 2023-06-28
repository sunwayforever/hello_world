#include <stdio.h>

__global__ void Rot13(char* text) {
    size_t id = threadIdx.x;
    char c = text[id];
    text[id] = (c - 1 / (~(~c | 32) / 13 * 2 - 11) * 13);
}

void rot13(char* text) {
    int N = strlen(text);
    char* d_text;
    cudaMalloc(&d_text, N);
    cudaMemcpy(d_text, text, N, cudaMemcpyHostToDevice);
    Rot13<<<1, N>>>(d_text);
    cudaMemcpy(text, d_text, N, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    char text[] = "Hello World";
    rot13(text);
    printf("%s\n", text);
    rot13(text);
    printf("%s\n", text);
    return 0;
}
