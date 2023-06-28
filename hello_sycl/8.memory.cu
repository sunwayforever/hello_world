#include <stdio.h>

#include <array>
// NOTE: constant memory, 必须声明为全局变量, 且长度为常量,
// 使用 __constant__ float* const_acc; 会出错
__constant__ float const_acc[10];
// NOTE: global memory, 必须声明为全局变量, 且长度为常量, 通过 cudaMalloc 可以没
// 有这个限制
__device__ float global_acc_symbol[10];
// NOTE: local memory, 可以声明为全局变量或在 kernel 中声明
__shared__ float local_acc[5];

__global__ void Dummy(float* global_acc) {
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    // NOTE: private memory
    float private_acc[1] = {const_acc[global_id]};
    global_acc[global_id] += private_acc[0] * 2;
    global_acc_symbol[global_id] = global_acc[global_id];
}

int main(int argc, char* argv[]) {
    std::array<float, 10> a = {1.0, 2.0, 3.0, 4.0, 5.0,
                               6.0, 7.0, 8.0, 9.0, 10.0};
    // NOTE: global memory, cudaMalloc 只能用来分配 global memory
    float* global_acc;
    cudaMalloc(&global_acc, sizeof(float) * 10);
    cudaMemcpy(
        global_acc, a.data(), sizeof(float) * 10, cudaMemcpyHostToDevice);
    // 对于非 cudaMalloc 分配的 memory (constant, device), 需要用
    // cudaMemcpy{From,To}Symbol 来读写
    cudaMemcpyToSymbol(global_acc_symbol, a.data(), sizeof(float) * 10);
    cudaMemcpyToSymbol(const_acc, a.data(), sizeof(float) * 10);
    // NOTE: local memory 无法通过 host访问
    // cudaMemcpyToSymbol(local_acc, a.data(), sizeof(float) * 5);

    Dummy<<<2, 5>>>(global_acc);

    cudaMemcpy(
        a.data(), global_acc, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");

    cudaMemcpyFromSymbol(a.data(), global_acc_symbol, sizeof(float) * 10);

    for (int i = 0; i < 10; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
    return 0;
}
