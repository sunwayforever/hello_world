all:native portable msa

MIPS_CC ?= mips64el-linux-gnuabi64-gcc
ARM_CC ?= aarch64-linux-gnu-gcc
CC ?= gcc

CFLAGS := -MMD -O2 -I.
LDFLAGS := -L. -static
LDLIBS := -lm

SRC := $(shell find ./ -name test_*.c -printf '%P\n')
NATIVE_APP := $(patsubst %.c,%_native.elf,${SRC})
PORTABLE_APP := $(patsubst %.c,%_portable.elf,${SRC})
MSA_APP := $(patsubst %.c,%_msa.elf,${SRC})

TEST_NATIVE_APP := $(patsubst %, run_%,${NATIVE_APP})
TEST_PORTABLE_APP := $(patsubst %, run_%,${PORTABLE_APP})
TEST_MSA_APP := $(patsubst %, run_%,${MSA_APP})

native: ${NATIVE_APP}
portable: ${PORTABLE_APP}
msa: ${MSA_APP}

DEP := $(NATIVE_APP:.elf=.d)
DEP += $(PORTABLE_APP:.elf=.d)
DEP += $(MSA_APP:.elf=.d)
-include ${DEP}

%_native.elf:%.c
	${ARM_CC} ${CFLAGS} -o $@ $< ${LDFLAGS} ${LDLIBS}
%_portable.elf:%.c
	${CC} ${CFLAGS} -Ineon_portable -o $@ $< ${LDFLAGS} ${LDLIBS}

# NOTE: use mips little endian, so that endianess problem won't bother
# us. e.g. you will run into endianess problem when implementing neon vcreate_s8
# with __msa_ld_b
%_msa.elf:%.c
	${MIPS_CC} -mmsa -march=loongson3a -Ineon_msa \
	-Ineon_portable  ${CFLAGS}  -o $@ $< ${LDFLAGS} ${LDLIBS}


${TEST_PORTABLE_APP}:run_%:%
	./$<

${TEST_NATIVE_APP}:run_%:%
	qemu-aarch64 ./$<

# NOTE: latest qemu is needed to support Loongson-3A4000
${TEST_MSA_APP}:run_%:%
	qemu-mips64el --cpu Loongson-3A4000 ./$<

run_native: ${TEST_NATIVE_APP}
run_portable: ${TEST_PORTABLE_APP}
run_msa: ${TEST_MSA_APP}
run: run_native run_portable run_msa

clean:
	@rm -f ${NATIVE_APP} ${PORTABLE_APP} ${MSA_APP} ${DEP}

.PHONY: clean run native portable msa
