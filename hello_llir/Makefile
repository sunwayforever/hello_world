all: dump

A_SRC := $(shell find ./ -name "*.cc" -printf '%P\n')
A_ELF := $(patsubst %.cc,%.elf,${A_SRC})
B_LL := $(patsubst %.cc,%.ll,${A_SRC})
B_OBJ := $(patsubst %.ll,%.o,${B_LL})
B_DUMP := $(patsubst %.o,%.dump,${B_OBJ})

${A_ELF}:%.elf:%.cc
	clang++-16 $< `llvm-config-16 --cxxflags --ldflags --system-libs --libs core` -o $@

${B_LL}:FORCE
${B_LL}:%.ll:%.elf
	./$< > $@

${B_OBJ}:%.o:%.ll
	clang-16 $< -c -o $@

${B_DUMP}:%.dump:%.o
	objdump -d $< > $@

dump:${B_DUMP}

clean:
	rm -f ${A_ELF} ${B_LL} ${B_OBJ} ${B_DUMP}

FORCE:
