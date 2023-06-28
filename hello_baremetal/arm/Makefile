OBJ=test.o startup.o
CFLAGS= -mcpu=arm926ej-s
ASFLAGS = -mcpu=arm926ej-s

CC=arm-none-eabi-gcc -g
AS=arm-none-eabi-as
LD=arm-none-eabi-ld
OBJCOPY=arm-none-eabi-objcopy
# CC=mips-img-elf-gcc -g
# AS=mips-img-elf-as
# LD=mips-img-elf-ld
# OBJCOPY=mips-img-elf-objcopy

all: test.bin

test.elf:${OBJ}
	${CC}  $^ -o $@ -v --specs=nosys.specs ${CFLAGS} -T test.ld

test.bin:test.elf
	${OBJCOPY} -O binary $^ $@

clean:
	-rm ${OBJ}
	-rm test.bin
	-rm test.elf

run:
	qemu-system-arm -M versatilepb -m 128M -nographic -kernel test.bin
