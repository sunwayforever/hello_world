#include <stdio.h>

#define N 1024
__global__ void ArraySum(int* dev_arr, int stride) {
    int id = 2 * stride * threadIdx.x;
    if (id < N) {
        dev_arr[id] = dev_arr[id] + dev_arr[id + stride];
    }
}

#define NN 1024
__global__ void ArraySumMoreData(int* dev_arr, int stride) {
    int linear_id = threadIdx.x + blockDim.x * blockIdx.x;
    int id = 2 * stride * linear_id;
    if (id < NN) {
        dev_arr[id] = dev_arr[id] + dev_arr[id + stride];
    }
}

void array_sum_stride() {
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        printf("%d ", arr[i]);
    }
    printf("\n");

    int* dev_arr;
    cudaMalloc(&dev_arr, sizeof(arr));
    cudaMemcpy(dev_arr, arr, sizeof(arr), cudaMemcpyHostToDevice);
    for (int stride = 1; stride < N; stride *= 2) {
        ArraySum<<<1, N>>>(dev_arr, stride);
    }
    cudaMemcpy(&arr, dev_arr, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", arr[0]);
}

void array_sum_stride_more_data() {
    int arr[NN];
    for (int i = 0; i < NN; i++) {
        arr[i] = i + 1;
        printf("%d ", arr[i]);
    }
    printf("\n");

    int* dev_arr;
    cudaMalloc(&dev_arr, sizeof(arr));
    cudaMemcpy(dev_arr, arr, sizeof(arr), cudaMemcpyHostToDevice);
    for (int stride = 1; stride < NN; stride *= 2) {
        ArraySumMoreData<<<int(NN / 1024) + 1, 1024>>>(dev_arr, stride);
    }
    cudaMemcpy(&arr, dev_arr, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", arr[0]);
}

int main(int argc, char* argv[]) {
    array_sum_stride();
    array_sum_stride_more_data();
    return 0;
}
