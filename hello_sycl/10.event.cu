#include "stdio.h"

__global__ void kernel_dummy_1() { printf("kernel_dummy_1\n"); }
__global__ void kernel_dummy_2() { printf("kernel_dummy_2\n"); }
__global__ void kernel_dummy_3() { printf("kernel_dummy_3\n"); }
__global__ void kernel_dummy_4() { printf("kernel_dummy_4\n"); }

int main(int argc, char *argv[]) {
    cudaStream_t stream_1;
    cudaStream_t stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    kernel_dummy_1<<<1, 1, 0, stream_1>>>();

    cudaEventRecord(start_event, stream_1);
    kernel_dummy_2<<<1, 1, 0, stream_1>>>();
    cudaEventRecord(stop_event, stream_1);

    kernel_dummy_3<<<1, 1, 0, stream_2>>>();

    float runtime = 0.0;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&runtime, start_event, stop_event);
    printf("runtime %f\n", runtime);

    kernel_dummy_4<<<1, 1, 0, stream_2>>>();

    cudaDeviceSynchronize();
    return 0;
}
