all: test

TEST_BIN := test.elf

CC=gcc
CXX=g++

GCCPLUGINS_DIR := $(shell ${CC} -print-file-name=plugin)
PLUGIN_CXXFLAGS := -I$(GCCPLUGINS_DIR)/include -fPIC -fno-rtti -O2

PLUGIN_SRC := callback.cc ../hello_plugin.cc
PLUGIN_OBJ := $(patsubst %.cc,%.o,${PLUGIN_SRC})

$(PLUGIN_OBJ):CXXFLAGS := ${PLUGIN_CXXFLAGS}

PLUGIN := libhello.so
${PLUGIN}: $(PLUGIN_OBJ)
	${CC} -shared $^ -o $@ -lstdc++

#-------------------------------------------
TEST_CFLAGS := -O0 -g3 -fplugin=./${PLUGIN} -fplugin-arg-libhello-count=1
TEST_CXXFLAGS := -O0 -g3 -fplugin=./${PLUGIN} -fplugin-arg-libhello-count=1

TEST_SRC := $(wildcard test*.c) + $(wildcard test*.cc)
TEST_OBJ := $(patsubst %.cc,%.o,$(patsubst %.c,%.o,${TEST_SRC}))

${TEST_OBJ}:CFLAGS:=${TEST_CFLAGS}
${TEST_OBJ}:CXXFLAGS:=${TEST_CXXFLAGS}

${TEST_OBJ}:${PLUGIN} FORCE

test:${TEST_OBJ}

FORCE:

${TEST_BIN}:${TEST_OBJ}
	${CC} $< -o $@ ${LDFLAGS} ${LDLIBS}

run: ${TEST_BIN}
	- ./${TEST_BIN}

clean:
	rm -f ${TEST_OBJ} ${PLUGIN} ${PLUGIN_OBJ} ${TEST_BIN}

.PHONY: test clean FORCE
