// 2021-09-17 18:20
#ifndef TVM_RUNTIME_H
#define TVM_RUNTIME_H

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

tvm::runtime::Module* tvm_runtime_create();

#endif  // TVM_RUNTIME_H
