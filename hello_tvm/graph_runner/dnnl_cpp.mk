USE_DNNL=1
include cpp.mk
LDLIBS += -ldnnl
cpp_runtime/kws: ${TVM_ROOT}/src/runtime/contrib/dnnl/dnnl.o
