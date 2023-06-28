#include "mspace_config.h"
#include "mspace_malloc.h"

extern void malloc_benchmark();

int main(int argc, char *argv[]) {
    init_spaces();
    malloc_benchmark();
}
