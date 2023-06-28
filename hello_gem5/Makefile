all:build
CFLAGS := -O0
CFLAGS_PROFILE_USE := -fbranch-probabilities -fenable-rtl-bbro
CFLAGS_SCHED := -fenable-rtl-sched1 -fenable-rtl-sched2
LDFLAGS := -L.
LDLIBS := -lm

SRC := $(wildcard *.c)
BENCHMARKS :=$(patsubst %.c, %, ${SRC})

X86_ELF := $(patsubst %.c, %.0.X86.elf,${SRC})
X86_SCHED_ELF := $(patsubst %.c, %.1.X86.elf,${SRC})
X86_BRANCH_ELF := $(patsubst %.c, %.2.X86.elf,${SRC})

RISCV_ELF := $(patsubst %.c, %.0.Riscv.elf,${SRC})
RISCV_SCHED_ELF := $(patsubst %.c, %.1.Riscv.elf,${SRC})
RISCV_BRANCH_ELF := $(patsubst %.c, %.2.Riscv.elf,${SRC})

ARM_ELF := $(patsubst %.c, %.0.Arm.elf,${SRC})
ARM_SCHED_ELF := $(patsubst %.c, %.1.Arm.elf,${SRC})
ARM_BRANCH_ELF := $(patsubst %.c, %.2.Arm.elf,${SRC})

# --------------------------------------------------------
${X86_ELF}:%.0.X86.elf:%.c
	gcc ${CFLAGS} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static

${X86_SCHED_ELF}:%.1.X86.elf:%.c
	gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ \
		$^ ${LDFLAGS} ${LDLIBS} -static
${X86_BRANCH_ELF}:%.2.X86.elf:%.c
	gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static -fprofile-generate -fprofile-dir=/tmp/$@; \
	./$@; \
	gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static ${CFLAGS_PROFILE_USE} -fprofile-dir=/tmp/$@; \
	rm -rf /tmp/$@
# --------------------------------------------------------
${RISCV_ELF}:%.0.Riscv.elf:%.c
	riscv64-linux-gnu-gcc ${CFLAGS} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static

${RISCV_SCHED_ELF}:%.1.Riscv.elf:%.c
	riscv64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ \
		$^ ${LDFLAGS} ${LDLIBS} -static

${RISCV_BRANCH_ELF}:%.2.Riscv.elf:%.c
	riscv64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static -fprofile-generate -fprofile-dir=/tmp/$@; \
	qemu-riscv64 ./$@; \
	riscv64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static ${CFLAGS_PROFILE_USE} -fprofile-dir=/tmp/$@; \
	rm -rf /tmp/$@

${RISCV_O2_ELF}:%.3.Riscv.elf:%.c
	riscv64-linux-gnu-gcc -O2 -o $@ $^ ${LDFLAGS} ${LDLIBS} -static
# --------------------------------------------------------
${ARM_ELF}:%.0.Arm.elf:%.c
	aarch64-linux-gnu-gcc ${CFLAGS} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static

${ARM_SCHED_ELF}:%.1.Arm.elf:%.c
	aarch64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ \
		$^ ${LDFLAGS} ${LDLIBS} -static

${ARM_BRANCH_ELF}:%.2.Arm.elf:%.c
	aarch64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static -fprofile-generate -fprofile-dir=/tmp/$@; \
	qemu-aarch64 ./$@; \
	aarch64-linux-gnu-gcc ${CFLAGS} ${CFLAGS_SCHED} -o $@ $^ ${LDFLAGS} ${LDLIBS} -static ${CFLAGS_PROFILE_USE} -fprofile-dir=/tmp/$@; \
	rm -rf /tmp/$@

build:${X86_ELF} ${X86_SCHED_ELF} ${X86_BRANCH_ELF} ${RISCV_ELF} ${RISCV_SCHED_ELF} ${RISCV_BRANCH_ELF} ${ARM_ELF} ${ARM_SCHED_ELF} ${ARM_BRANCH_ELF}

${BENCHMARKS}:%:
	./gem5.py $@

benchmark:${BENCHMARKS}

clean:
	rm -f *.elf

.PHONY: build clean benchmark ${BENCHMARKS}
