#include <stdio.h>

__global__ void Range() { printf("%d %d\n", threadIdx.x, threadIdx.y); }
__global__ void NdRange() {
    printf(
        "global:%d group:%d local:%d\n", blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.x, threadIdx.x);
}
__global__ void NdRange2() {
    int local_id_x = threadIdx.x;
    int local_id_y = threadIdx.y;
    // int local_linear_id = local_id_y * blockDim.x + local_id_x;
    int local_linear_id = local_id_x * blockDim.y + local_id_y;

    int group_id_x = blockIdx.x;
    int group_id_y = blockIdx.y;
    int group_linear_id = group_id_y * gridDim.x + group_id_x;

    int global_id_x = gridDim.x * blockIdx.x + threadIdx.x;
    int global_id_y = gridDim.y * blockIdx.y + threadIdx.y;
    // int global_linear_id = global_id_y * gridDim.x * blockDim.x +
    // global_id_x;
    int global_linear_id = global_id_x * gridDim.y * blockDim.y + global_id_y;

    printf(
        "global:%d global[0]:%d global[1]:%d | group:%d "
        "group[0]:%d group[1]:%d | local:%d "
        "local[0]:%d "
        "local[1]:%d\n",
        global_linear_id, global_id_x, global_id_y, group_linear_id, group_id_x,
        group_id_y, local_linear_id, local_id_x, local_id_y);
}
int main(int argc, char* argv[]) {
    Range<<<1, dim3(2, 5)>>>();
    cudaDeviceSynchronize();
    printf("nd_range\n");
    NdRange<<<2, 5>>>();
    cudaDeviceSynchronize();
    NdRange2<<<dim3(2, 1), dim3(2, 5)>>>();
    cudaDeviceSynchronize();
    return 0;
}
