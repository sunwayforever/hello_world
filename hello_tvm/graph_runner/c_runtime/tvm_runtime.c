// 2021-08-09 11:02
#include "tvm_runtime.h"

#define CRT_MEMORY_NUM_PAGES 16384
#define CRT_MEMORY_PAGE_SIZE_LOG2 10

extern unsigned char kws_graph_json[];
extern unsigned int kws_graph_json_len;

extern unsigned char kws_params_bin[];
extern unsigned int kws_params_bin_len;

static uint8_t
    g_crt_memory[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];
static MemoryManagerInterface* g_memory_manager;

void TVMLogf(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
    fprintf(stderr, "TVMPlatformAbort: %d\n", error_code);
    exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(
    size_t num_bytes, DLDevice dev, void** out_ptr) {
    *out_ptr = malloc(num_bytes);
    memset(*out_ptr, 0xff, num_bytes);
    if (out_ptr == NULL) return kTvmErrorPlatformNoMemory;
    return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    free(ptr);
    return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStart() {
    return kTvmErrorFunctionCallNotImplemented;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
    return kTvmErrorFunctionCallNotImplemented;
}

TVMGraphExecutor* tvm_runtime_create() {
    DLDevice dev;
    dev.device_type = (DLDeviceType)kDLCPU;
    dev.device_id = 0;

    // get pointers
    PageMemoryManagerCreate(
        &g_memory_manager, g_crt_memory, sizeof(g_crt_memory),
        CRT_MEMORY_PAGE_SIZE_LOG2);
    TVMInitializeRuntime();
    TVMPackedFunc pf;
    TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
    TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args);
    TVMPackedFunc_Call(&pf);

    TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

    TVMGraphExecutor* graph_executor = NULL;
    TVMGraphExecutor_Create(
        (const char*)kws_graph_json, mod_syslib, &dev, &graph_executor);
    TVMGraphExecutor_LoadParams(
        graph_executor, (const char*)kws_params_bin, kws_params_bin_len);

    return graph_executor;
}

// https://gustedt.wordpress.com/2010/11/29/myth-and-reality-about-inline-in-c99/
extern inline TVMModuleHandle TVMArgs_AsModuleHandle(
    const TVMArgs* args, size_t index);
