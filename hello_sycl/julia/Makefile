ALL:julia.sycl.elf julia.cu.elf

.PHONY: ALL clean

SYCL_CPPFLAGS := -I ~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/include/
SYCL_LDFLAGS := -L ~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/lib/ -Wl,-rpath,/home/sunway/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/lib/
SYCL_LDLIBS := -lComputeCpp -lOpenCL

CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64
CUDA_LDLIBS := -lcudart

CPPFLAGS := -I/home/sunway/download/Cinder/include -MMD -O3
LDLIBS := /home/sunway/download/Cinder/lib/linux/x86_64/ogl/Debug/libcinder.a \
		  -ldl -lOpenGL -lcurl -lpthread -lX11 -lfontconfig -lXrandr -lXinerama -lXcursor \

CPPFLAGS += ${SYCL_CPPFLAGS}
LDFLAGS += ${SYCL_LDFLAGS} ${CUDA_LDFLAGS}
LDLIBS += ${SYCL_LDLIBS} ${CUDA_LDLIBS}

SRC:=$(wildcard *.cc)
OBJ:=$(patsubst %.cc,%.o,${SRC})

SYCL_SRC:=$(wildcard sycl/*.cc)
SYCL_HEADER:=$(patsubst %.cc,%.sycl,${SYCL_SRC})
SYCL_OBJ:=$(patsubst %.cc,%.o,${SYCL_SRC})

CUDA_KERNEL_SRC:=$(wildcard cuda/*.cu)
CUDA_SRC := $(wildcard cuda/*.cc)
CUDA_OBJ := $(patsubst %.cu,%.o,${CUDA_KERNEL_SRC}) $(patsubst %.cc,%.o,${CUDA_SRC})

ALL_OBJ := ${OBJ} ${SYCL_OBJ} ${CUDA_OBJ}
DEPS := $(ALL_OBJ:.o=.d)
-include $(DEPS)

%.sycl:%.cc
	~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/bin/compute $^ ${SYCL_CPPFLAGS} -sycl -c -o $@

%.o:%.cu
	nvcc -c $^ -o $@

julia.sycl.elf:${OBJ} ${SYCL_HEADER} ${SYCL_OBJ}
	g++ ${OBJ} ${SYCL_OBJ} -o $@ ${LDFLAGS} ${LDLIBS} -include ${SYCL_HEADER}

julia.cu.elf:${OBJ} ${CUDA_OBJ}
	g++ $^ -o $@ ${LDFLAGS} ${LDLIBS}

clean:
	-rm -f julia.sycl.elf
	-rm -f julia.cu.elf
	-rm -f ${ALL_OBJ}
	-rm -f ${DEPS}
	-rm -f ${SYCL_HEADER}
