#include <assert.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#define WORKING_SET_SIZE 1024
#define MIN_ALLOCATION_SIZE 4
#define MAX_ALLOCATION_SIZE 32768

#define MAX_ITER 1000000

/* Get a random block size with an inverse square distribution.  */
static unsigned int get_block_size(unsigned int rand_data) {
    /* Inverse square.  */
    const float exponent = -2;
    const float dist_min = MIN_ALLOCATION_SIZE;
    const float dist_max = MAX_ALLOCATION_SIZE;

    float min_pow = powf(dist_min, exponent + 1);
    float max_pow = powf(dist_max, exponent + 1);
    float r = (float)rand_data / RAND_MAX;

    return (unsigned int)powf(
        (max_pow - min_pow) * r + min_pow, 1 / (exponent + 1));
}

#define NUM_BLOCK_SIZES 8000
#define NUM_OFFSETS ((WORKING_SET_SIZE)*4)

static unsigned int random_block_sizes[NUM_BLOCK_SIZES];
static unsigned int random_offsets[NUM_OFFSETS];

static void init_random_values(void) {
    srand(1);
    for (size_t i = 0; i < NUM_BLOCK_SIZES; i++) {
        random_block_sizes[i] = get_block_size(rand());
    }

    for (size_t i = 0; i < NUM_OFFSETS; i++) {
        random_offsets[i] = rand() % WORKING_SET_SIZE;
    }
}

static unsigned int get_random_block_size(unsigned int *idx) {
    *idx = *idx >= NUM_BLOCK_SIZES - 1 ? 0 : (*idx) + 1;
    return random_block_sizes[*idx];
}

static unsigned int get_random_offset(unsigned int *idx) {
    *idx = *idx >= NUM_OFFSETS - 1 ? 0 : (*idx) + 1;
    return random_offsets[*idx];
}

static void malloc_benchmark_loop(void **ptr_arr) {
    unsigned int offset_idx = 0, block_idx = 0;
    size_t iters = 0;

    while (iters < MAX_ITER) {
        unsigned int offset = get_random_offset(&offset_idx);
        unsigned int block = get_random_block_size(&block_idx);
        free(ptr_arr[offset]);
        ptr_arr[offset] = malloc(block);
        assert(ptr_arr[offset] != NULL);
        // memset(ptr_arr[offset], 1, block);
        iters++;
    }
}

void malloc_benchmark() {
    init_random_values();
    void *working_set[WORKING_SET_SIZE];
    memset(working_set, 0, sizeof(working_set));
    malloc_benchmark_loop(working_set);
}
