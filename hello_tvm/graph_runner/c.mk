all: c_runtime/kws

RUNTIME=c

include libkws.mk

CRT_ROOT=${TVM_ROOT}/build/standalone_crt
CRT_SRCS = $(shell find $(CRT_ROOT))

crt/libcommon.a: $(CRT_SRCS)
	cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath .)/crt CRT_CONFIG=$(abspath crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" common

crt/libmemory.a: $(CRT_SRCS)
	cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath .)/crt CRT_CONFIG=$(abspath crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" memory

crt/libgraph_executor.a: $(CRT_SRCS)
	$(QUIET)cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath .)/crt CRT_CONFIG=$(abspath crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" graph_executor

C_RUNTIME_OBJ=c_runtime/kws.o c_runtime/tvm_runtime.o test/test_xiaoai.o test/test_unknown.o

c_runtime/kws: libkws.a
c_runtime/kws: ${C_RUNTIME_OBJ} crt/libmemory.a crt/libcommon.a crt/libgraph_executor.a

run:c_runtime/kws
	./c_runtime/kws 2>/dev/null

clean:common_clean
	-rm ${C_RUNTIME_OBJ}
	-rm c_runtime/kws
	-rm -rf crt
