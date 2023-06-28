#include <stdio.h>

__global__ void ArraySum(int* dev_arr, int* result) {
    int id = threadIdx.x;
    atomicAdd(result, dev_arr[id]);
}

#define N 100
int main(int argc, char* argv[]) {
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        printf("%d ", arr[i]);
    }
    printf("\n");

    int result = 0;
    int *dev_arr, *dev_result;
    cudaMalloc(&dev_arr, sizeof(arr));
    cudaMalloc(&dev_result, sizeof(int));
    cudaMemcpy(dev_arr, arr, sizeof(arr), cudaMemcpyHostToDevice);
    ArraySum<<<1, N>>>(dev_arr, dev_result);
    cudaMemcpy(&result, dev_result, sizeof(result), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", result);
    return 0;
}
