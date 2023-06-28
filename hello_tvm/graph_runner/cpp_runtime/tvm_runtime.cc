#include "tvm_runtime.h"

extern unsigned char kws_graph_json[];
extern unsigned int kws_graph_json_len;

extern unsigned char kws_params_bin[];
extern unsigned int kws_params_bin_len;

tvm::runtime::Module* tvm_runtime_create() {
    const std::string json_data(
        &kws_graph_json[0], &kws_graph_json[0] + kws_graph_json_len);
    tvm::runtime::Module mod_syslib =
        (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
    int device_type = kDLCPU;
    int device_id = 0;

    tvm::runtime::Module mod =
        (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
            json_data, mod_syslib, device_type, device_id);
    TVMByteArray params;
    params.data = reinterpret_cast<const char*>(&kws_params_bin[0]);
    params.size = kws_params_bin_len;
    mod.GetFunction("load_params")(params);
    return new tvm::runtime::Module(mod);
}
