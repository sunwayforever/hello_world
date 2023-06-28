#include "tvm_runtime.h"
extern unsigned char test_xiaoai[];
extern unsigned int test_xiaoai_len;

extern unsigned char test_unknown[];
extern unsigned int test_unknown_len;

int main(int argc, char* argv[]) {
    tvm::runtime::Module* handle = tvm_runtime_create();

    auto set_input = handle->GetFunction("set_input");
    auto get_output = handle->GetFunction("get_output");
    auto run = handle->GetFunction("run");

    float input_data[1 * 99 * 12] = {0};

    DLTensor input;
    input.data = input_data;
    DLDevice dev = {kDLCPU, 0};
    input.device = dev;
    input.ndim = 3;
    DLDataType dtype = {kDLFloat, 32, 1};
    input.dtype = dtype;
    int64_t input_shape[] = {1, 99, 12};
    input.shape = input_shape;
    input.strides = NULL;
    input.byte_offset = 0;

    float output_data[1] = {0};
    DLTensor output;
    output.data = output_data;
    DLDevice out_dev = {kDLCPU, 0};
    output.device = out_dev;
    output.ndim = 2;
    DLDataType out_dtype = {kDLFloat, 32, 1};
    output.dtype = out_dtype;
    int64_t output_shape[] = {1, 1};
    output.shape = output_shape;
    output.strides = NULL;
    output.byte_offset = 0;

    handle->GetFunction("set_input")("input_1", &input);

    int STEP = 99 * 12 * sizeof(float) / sizeof(char);

#define TEST(data, len)                             \
    {                                               \
        unsigned char* mfcc = (unsigned char*)data; \
        int n = len / STEP;                         \
        for (int i = 0; i < n; i++) {               \
            memcpy(input_data, mfcc, STEP);         \
            set_input("input_1", &input);           \
            mfcc += STEP;                           \
            run();                                  \
            get_output(0, &output);                 \
            printf("output: %f\n", output_data[0]); \
        }                                           \
    }

    printf("\n------test xiaoai------\n");
    TEST(test_xiaoai, test_xiaoai_len);

    printf("\n------test unknown------\n");
    TEST(test_unknown, test_unknown_len);
    return 0;
}
