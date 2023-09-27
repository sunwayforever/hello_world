# build pocl:
# cd <path_to_pocl>
# mkdir build
# cd build
# cmake .. -DENABLE_ICD=0 && make
POCL := /home/sunway/source/pocl

SRC := $(wildcard *.c)
OBJ := $(patsubst %.c,%.o,${SRC})

CPPFLAGS := -DCL_HPP_TARGET_OPENCL_VERSION=300 \
			-DCL_TARGET_OPENCL_VERSION=300 \
			-DCL_USE_DEPRECATED_OPENCL_1_0_APIS \
			-DCL_USE_DEPRECATED_OPENCL_1_1_APIS \
			-DCL_USE_DEPRECATED_OPENCL_1_2_APIS \
			-DCL_USE_DEPRECATED_OPENCL_2_0_APIS \
			-DCL_USE_DEPRECATED_OPENCL_2_1_APIS \
			-DCL_USE_DEPRECATED_OPENCL_2_2_APIS \
			-I${POCL}/include -I${POCL}/build -I${POCL}/include/hpp -I${POCL}/poclu

CFLAGS := -g -O0

LDFLAGS := -Wl,-rpath,${POCL}/build/lib/CL:/usr/lib/llvm-16/lib
LDLIBS := ${POCL}/build/poclu/libpoclu.a \
			${POCL}/build/lib/CL/libOpenCL.so.2.12.0 \
			/usr/lib/llvm-16/lib/libclang-cpp.so \
			/usr/lib/llvm-16/lib/libLLVM-16.so -lrt -lm -ldl -lm -pthread -ldl

.PHONY: all clean

APP := $(patsubst %.c,%.elf,${SRC})
RUN_APP := $(patsubst %.elf, run_%, ${APP})

all: ${APP}

${APP}:%.elf:%.o
	gcc $< -o $@ ${LDFLAGS} ${LDLIBS}

${RUN_APP}:run_%:%.elf
	POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1	POCL_DEVICES="basic" POCL_DEBUG=all POCL_BUILDING=1 ./$<

clean:
	-rm -f *.elf *.o
