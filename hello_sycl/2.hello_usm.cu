#include <stdio.h>

__global__ void VecAdd(float4* a, float4* b, float4* c) {
    // NOTE: float4 等类型可以用 x,y,z,w 表示 0,1,2,3, float4 是
    // sycl::vec<float,4> 的别名, sycl::vec 可以用如下的别名作为下标:
    // x,y,z,w (0,1,2,3)
    // r,g,b,a(0,1,2,3)
    // s0...s9,sA...sF (0...15)
    c[0].x = a[0].x + b[0].x;
    c[0].y = a[0].y + b[0].y;
    c[0].z = a[0].z + b[0].z;
    c[0].w = a[0].w + b[0].w;
}

void hello_usm() {
    float4 a = {1.0, 2.0, 3.0, 4.0};
    float4 b = {1.0, 2.0, 3.0, 4.0};
    float4 c = {0.0, 0.0, 0.0, 0.0};

    float4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(float4));
    cudaMalloc(&d_b, sizeof(float4));
    cudaMalloc(&d_c, sizeof(float4));

    cudaMemcpy(d_a, &a, sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float4), cudaMemcpyHostToDevice);

    VecAdd<<<1, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, sizeof(float4), cudaMemcpyDeviceToHost);
    printf("%f %f %f %f\n", c.x, c.y, c.z, c.w);
}

void hello_usm_shared() {
    float4 a = {1.0, 2.0, 3.0, 4.0};
    float4 b = {1.0, 2.0, 3.0, 4.0};

    float4 *d_a, *d_b, *c;
    cudaMalloc(&d_a, sizeof(float4));
    cudaMalloc(&d_b, sizeof(float4));
    cudaMallocManaged(&c, sizeof(float4));

    cudaMemcpy(d_a, &a, sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float4), cudaMemcpyHostToDevice);

    VecAdd<<<1, 1>>>(d_a, d_b, c);

    cudaDeviceSynchronize();

    printf("%f %f %f %f\n", c->x, c->y, c->z, c->w);
}

int main(int argc, char* argv[]) {
    hello_usm();
    hello_usm_shared();
    return 0;
}
