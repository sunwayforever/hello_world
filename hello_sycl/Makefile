# 1. Get ComputeCPP from https://developer.codeplay.com/products/computecpp/ce/download/
# 2. apt install intel-opencl-icd
CUDA_SRC := $(wildcard *.cu)
NVCC := /opt/anaconda3/envs/cuda-11/bin/nvcc

SYCL_SRC := $(wildcard *.cc)
SYCL_CPPFLAGS := -I ~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/include/
SYCL_LDFLAGS := -L ~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/lib/ -Wl,-rpath,/home/sunway/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/lib/
SYCL_LDLIBS := -lComputeCpp -lOpenCL

.PHONY: ALL clean cuda sycl

SYCL_APP := $(patsubst %.cc,%.sycl.elf,${SYCL_SRC})
CUDA_APP := $(patsubst %.cu,%.cu.elf,${CUDA_SRC})

ALL: sycl cuda

cuda:${CUDA_APP}
sycl:${SYCL_APP}

# %.elf:%.cc
# 	~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/bin/compute $^ ${SYCL_CPPFLAGS} ${SYCL_LDFLAGS} ${SYCL_LDLIBS} -sycl-driver -o $@

%.sycl:%.cc
	~/program/ComputeCpp-CE-2.8.0-x86_64-linux-gnu/bin/compute $^ ${SYCL_CPPFLAGS} -sycl -c $^

%.sycl.elf: %.sycl %.cc
	g++ ${SYCL_CPPFLAGS} ${SYCL_LDFLAGS} -include $^ -o $@ ${SYCL_LDLIBS}

%.cu.elf: %.cu
	${NVCC} $^ -o $@

clean:
	-rm -f *.elf
	-rm -f *.sycl
