all: test_mspace.elf test_bump_pointer.elf test_pool.elf

SHELL := /bin/bash

CFLAGS := -Os -flto -DONLY_MSPACES -DNO_MALLOC_STATS -DNO_MALLINFO -DHAVE_MMAP=0 \
	-DNO_MSPACES_FOOTPRINT -DNO_MSPACES_REALLOC -DNO_MSPACES_IALLOC -DNO_MSPACES_MEMALIGN \
	-DNO_MSPACES_BULK_FREE -DNO_MSPACES_MALLOPT -MMD -m32

# CFLAGS += -DUSE_HXD_PREFIX

# for mspace_malloc hist debug
# CFLAGS += -DDEBUG_HIST
# LIB_OBJ += printf.o
# end

CC := gcc
# CC := mips-linux-musl-gcc
# CC := aarch64-linux-android23-clang
# CC := riscv64-linux-gnu-gcc

MSPACE_SRC := $(wildcard mspace/*.c)
MSPACE_TEST_SRC := $(wildcard mspace/test_*.c)
MSPACE_SRC := $(filter-out ${MSPACE_TEST_SRC},${MSPACE_SRC})
MSPACE_OBJ := $(patsubst %.c,%.o,${MSPACE_SRC})
MSPACE_TEST_OBJ := $(patsubst %.c,%.o,${MSPACE_TEST_SRC})
libmspace_malloc.a:${MSPACE_OBJ}

BUMP_POINTER_SRC := $(wildcard bump_pointer/*.c)
BUMP_POINTER_TEST_SRC := $(wildcard bump_pointer/test_*.c)
BUMP_POINTER_SRC := $(filter-out ${BUMP_POINTER_TEST_SRC},${BUMP_POINTER_SRC})
BUMP_POINTER_OBJ := $(patsubst %.c,%.o,${BUMP_POINTER_SRC})
BUMP_POINTER_TEST_OBJ := $(patsubst %.c,%.o,${BUMP_POINTER_TEST_SRC})
libbump_pointer_malloc.a:${BUMP_POINTER_OBJ}

POOL_SRC := $(wildcard pool/*.c)
POOL_TEST_SRC := $(wildcard pool/test_*.c)
POOL_SRC := $(filter-out ${POOL_TEST_SRC},${POOL_SRC})
POOL_OBJ := $(patsubst %.c,%.o,${POOL_SRC})
POOL_TEST_OBJ := $(patsubst %.c,%.o,${POOL_TEST_SRC})
libpool_malloc.a:${POOL_OBJ}

TEST_OBJ:=malloc_benchmark.o
%a: ${LIB_OBJ}
	ar rcs $@ $^

OBJ:=${MSPACE_OBJ} ${BUMP_POINTER_OBJ}\
	${POOL_OBJ} ${MSPACE_TEST_OBJ} ${BUMP_POINTER_TEST_OBJ} ${POOL_TEST_OBJ} ${LIB_OBJ} ${TEST_OBJ}

${OBJ}:Makefile

DEP := $(OBJ:.o=.d)
-include ${DEP}

test_mspace.elf: ${MSPACE_TEST_OBJ} libmspace_malloc.a
test_bump_pointer.elf: ${BUMP_POINTER_TEST_OBJ} libbump_pointer_malloc.a
test_pool.elf: ${POOL_TEST_OBJ} libpool_malloc.a

%.elf: ${TEST_OBJ}
	${CC} ${CFLAGS} $^ -o $@ -lm

test:test_mspace.elf test_bump_pointer.elf test_pool.elf
	time ./test_mspace.elf; \
	time ./test_pool.elf; \
	time ./test_bump_pointer.elf;

clean:
	rm -rf *.elf *.a ${OBJ} ${DEP}
