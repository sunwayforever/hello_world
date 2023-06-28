OBJ=test.o startup.o
CFLAGS= -mips32r6
ASFLAGS = -mips32r6

CC=mips-img-elf-gcc
AS=mips-img-elf-as
LD=mips-img-elf-ld
OBJCOPY=mips-img-elf-objcopy

all: test.elf

test.elf:${OBJ}
	${CC}  $^ -o $@ -v ${CFLAGS} --specs=nosys.specs -T test.ld

clean:
	-rm ${OBJ}
	-rm test.elf

run:
	qemu-system-mips -M malta -m 128M -nographic -kernel test.elf
