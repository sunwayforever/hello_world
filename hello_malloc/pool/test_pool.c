#include "pool_config.h"
#include "pool_malloc.h"

extern void malloc_benchmark();

int main(int argc, char *argv[]) {
    init_spaces();
    malloc_benchmark();
}
