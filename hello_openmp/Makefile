CFLAGS := -fopenmp -O2
SRC := $(wildcard *.c)
APP := $(patsubst %.c, %.elf,${SRC})

all:${APP}

%.elf: %.c
	gcc -o $@ $^ ${CFLAGS}

# `apt install gcc-offload-nvptx` is needed
8.target.elf: CFLAGS += -fcf-protection=none -foffload=-misa=sm_35 -fno-stack-protector

clean:
	-rm -f ${APP}
