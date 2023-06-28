#include <stdio.h>
#include <cuda_profiler_api.h>

#define N 1026

#define WGROUP_SIZE 32

__shared__ float local_mem[WGROUP_SIZE];

__global__ void ArraySum(int* global_mem, int len) {
    int local_id = threadIdx.x;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;

    local_mem[local_id] = 0;
    if ((2 * global_id) < len) {
        local_mem[local_id] = global_mem[2 * global_id];
    }
    if ((2 * global_id + 1) < len) {
        local_mem[local_id] += global_mem[2 * global_id + 1];
    }

    __syncthreads();

    for (int stride = 1; stride < WGROUP_SIZE; stride *= 2) {
        int idx = 2 * stride * local_id;
        if (idx < WGROUP_SIZE) {
            local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
        }
        __syncthreads();
    }
    if (local_id == 0) {
        global_mem[blockIdx.x] = local_mem[0];
    }
}

int main(int argc, char* argv[]) {
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        printf("%d ", arr[i]);
    }
    printf("\n");

    int* global_mem;
    cudaMalloc(&global_mem, sizeof(arr));
    cudaMemcpy(global_mem, arr, sizeof(arr), cudaMemcpyHostToDevice);
    int len = N;
    while (len != 1) {
        int n_wgroups = (len + 2 * WGROUP_SIZE - 1) / (2 * WGROUP_SIZE);
        ArraySum<<<n_wgroups, WGROUP_SIZE>>>(global_mem, len);
        len = n_wgroups;
    }

    cudaMemcpy(arr, global_mem, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", arr[0]);
    return 0;
}
