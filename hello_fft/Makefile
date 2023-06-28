all: hello_fft

LDLIBS := -lm
CFLAGS := -O2

SRC := $(wildcard *.c)
OBJ := $(patsubst %.c,%.o,${SRC})

${OBJ}:Makefile

hello_fft:${OBJ} kissfft/libkissfft-float.a

kissfft/libkissfft-float.a: FORCE
	make -C kissfft KISSFFT_STATIC=1

FORCE:

clean:
	rm hello_fft ${OBJ}

run:hello_fft
	./hello_fft
