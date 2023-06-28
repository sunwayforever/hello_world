all: cpp_runtime/kws

RUNTIME=c++
include libkws.mk
LDLIBS += -lstdc++ -lpthread -ldnnl

TVM_SRC= \
   ${TVM_ROOT}/src/runtime/metadata_module.cc \
   ${TVM_ROOT}/src/runtime/contrib/dnnl/dnnl_json_runtime.cc\
   ${TVM_ROOT}/src/runtime/graph_executor/graph_executor.cc \
   ${TVM_ROOT}/src/runtime/graph_executor/graph_executor_factory.cc \
   ${TVM_ROOT}/src/runtime/c_runtime_api.cc \
   ${TVM_ROOT}/src/runtime/container.cc \
   ${TVM_ROOT}/src/runtime/cpu_device_api.cc \
   ${TVM_ROOT}/src/runtime/file_utils.cc \
   ${TVM_ROOT}/src/runtime/library_module.cc \
   ${TVM_ROOT}/src/runtime/logging.cc \
   ${TVM_ROOT}/src/runtime/module.cc \
   ${TVM_ROOT}/src/runtime/ndarray.cc \
   ${TVM_ROOT}/src/runtime/object.cc \
   ${TVM_ROOT}/src/runtime/registry.cc \
   ${TVM_ROOT}/src/runtime/system_library.cc \
   ${TVM_ROOT}/src/runtime/thread_pool.cc \
   ${TVM_ROOT}/src/runtime/threading_backend.cc \
   ${TVM_ROOT}/src/runtime/workspace_pool.cc \

TVM_OBJ=$(patsubst %.cc,%.o,$(TVM_SRC))

CPP_RUNTIME_OBJ=cpp_runtime/kws.o cpp_runtime/tvm_runtime.o test/test_xiaoai.o test/test_unknown.o

cpp_runtime/kws:libkws.a
cpp_runtime/kws: ${CPP_RUNTIME_OBJ} ${TVM_OBJ}

run:cpp_runtime/kws
	./cpp_runtime/kws

clean:common_clean
	-rm ${CPP_RUNTIME_OBJ}
	-rm ${TVM_OBJ}
	-rm cpp_runtime/kws
