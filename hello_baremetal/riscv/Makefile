BIN :=main.elf

all:${BIN}

CFLAGS := -O0 -g3 -march=rv64g -mabi=lp64 -mcmodel=medany
LDFLAGS := -nostdlib -nostartfiles -Tlink.ld
CC :=riscv64-unknown-elf-gcc

SRC := $(wildcard *.c *.s)

${BIN}:${SRC}

%.elf:
	${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS} ${LDLIBS}

run:${BIN}
	spike -l ${BIN}

clean:
	rm -f ${BIN}

.PHONY: run clean
