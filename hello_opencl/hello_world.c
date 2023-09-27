#include <CL/cl.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VECTOR_SIZE 16

const char *saxpy_kernel =
    "__kernel                                   \n"
    "void saxpy_kernel(float alpha,     \n"
    "                  __global float *A,       \n"
    "                  __global float *B,       \n"
    "                  __global float *C)       \n"
    "{                                          \n"
    "    int index = get_global_id(0);          \n"
    "    C[index] = alpha* A[index] + B[index]; \n"
    "}                                          \n";

char **load_binary(char *fname, size_t *out_size) {
    FILE *fp = fopen(fname, "rb");
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char **bins = malloc(sizeof(char *));
    bins[0] = (char *)malloc(size);
    fread(bins[0], 1, size, fp);
    fclose(fp);
    *out_size = size;
    return bins;
}

void save_binary(char *fname, cl_program program) {
    size_t size = 0;
    clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL);
    size_t ret = 0;
    char *bins[1];
    bins[0] = (char *)malloc(size);
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL);
    FILE *fp = fopen(fname, "wb");
    fwrite(bins[0], 1, size, fp);
    fclose(fp);
}

int main(void) {
    int i;
    float alpha = 2.0;
    float *hostA = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *hostB = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *hostC = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    for (i = 0; i < VECTOR_SIZE; i++) {
        hostA[i] = i;
        hostB[i] = VECTOR_SIZE - i;
        hostC[i] = 0;
    }

    cl_uint n_platforms;
    clGetPlatformIDs(0, NULL, &n_platforms);
    assert(n_platforms == 1);

    cl_platform_id platform;
    clGetPlatformIDs(n_platforms, &platform, NULL);

    cl_uint n_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
    assert(n_devices == 1);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n_devices, &device, NULL);

    cl_context context;
    context = clCreateContext(NULL, n_devices, &device, NULL, NULL, NULL);

    cl_command_queue command_queue =
        clCreateCommandQueue(context, device, 0, NULL);

    cl_mem targetA = clCreateBuffer(
        context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, NULL);
    cl_mem targetB = clCreateBuffer(
        context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, NULL);
    cl_mem targetC = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(
        command_queue, targetA, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), hostA,
        0, NULL, NULL);
    clEnqueueWriteBuffer(
        command_queue, targetB, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), hostB,
        0, NULL, NULL);

#if 0
    size_t size;
    const unsigned char **bins =
        (const unsigned char **)load_binary("/tmp/1.o", &size);
    cl_program program =
        clCreateProgramWithBinary(context, 1, &device, &size, bins, NULL,
    NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
#else
    cl_program program = clCreateProgramWithSource(
        context, 1, (const char **)&saxpy_kernel, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    save_binary("/tmp/1.o", program);
#endif

    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", NULL);

    clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&targetA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&targetB);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&targetC);

    size_t global_size = VECTOR_SIZE;
    size_t local_size = VECTOR_SIZE / 2;
    clEnqueueNDRangeKernel(
        command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL,
        NULL);
    clEnqueueReadBuffer(
        command_queue, targetC, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), hostC,
        0, NULL, NULL);
    clFlush(command_queue);
    clFinish(command_queue);
    for (i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, hostA[i], hostB[i], hostC[i]);

    return 0;
}
