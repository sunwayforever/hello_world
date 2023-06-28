USE_DNNL?=0
ifeq (${USE_DNNL},1)
	build_libkws=python ./libkws.py --runtime=${RUNTIME} --dnnl
else
	build_libkws=python ./libkws.py --runtime=${RUNTIME}
endif

TVM_ROOT=/home/sunway/source/tvm

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O0 -fPIC -fshort-enums
CPPFLAGS = ${PKG_COMPILE_OPTS} \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-I${TVM_ROOT}/src/runtime/contrib/ \
	-I. \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

LDLIBS += -Wl,--whole-archive libkws.a -Wl,-no-whole-archive -lm

libkws.a:libkws.py
	${build_libkws}
	make CPPFLAGS="${CPPFLAGS}" -f /tmp/libkws/libkws.mk

common_clean:
	-rm -rf /tmp/libkws
	-rm libkws.a
