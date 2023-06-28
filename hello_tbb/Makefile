CFLAGS := -O0 -I/opt/intel/oneapi/tbb/latest/include
LDFLAGS := -L/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8/ \
		   -Wl,-rpath,/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8/
LDLIBS := -ltbb -ltbbmalloc

SRC := $(wildcard *.cc)
APP := $(patsubst %.cc, %.elf,${SRC})

all:${APP}

%.elf: %.cc
	g++ -o $@ $^ ${CFLAGS} ${LDFLAGS} ${LDLIBS}

clean:
	-rm -f ${APP}
