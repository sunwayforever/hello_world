TEST:=$(patsubst %.c,%.elf,$(wildcard test_*.c))

LDLIBS:=-lbfd
CFLAGS:=-g

all:${TEST}

%.elf:%.c
	${CC} ${CFLAGS} -o $@ $^ ${LDLIBS}

test_write_output.elf: test_write_reloc.o test_write_section.o

test_write_section.o test_write_reloc.o:test_write.elf
	./test_write.elf

clean:
	-rm -rf ${TEST}
	-rm test_write*.o
