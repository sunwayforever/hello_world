CC := riscv64-linux-gnu-gcc

all: hello.elf

hello.elf: hello.s
	${CC} -T link.ld -o hello.elf hello.s -nostdlib -g

clean:
	rm -f *.o hello.elf

run: hello.elf
	./run.sh
