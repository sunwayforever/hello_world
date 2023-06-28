#include "pool_malloc.h"

#include <stdint.h>

extern size_t BIN_SIZE;
extern void* POOLS[];
extern void* BUFFERS[];
extern size_t BUFFER_CAPACITIES[];
extern size_t BUFFER_SIZES[];
extern size_t BUFFER_COUNTS[];
extern int N_BUFFER;

struct Chunk {
    struct Chunk* next;
};

typedef struct {
    struct Chunk* free_list;
    size_t size;
} Pool;

Pool* init_pool(void* base, size_t capacity, size_t size, size_t count);

void init_spaces() {
    for (int i = 0; i < N_BUFFER; i++) {
        init_pool(
            BUFFERS[i], BUFFER_CAPACITIES[i], BUFFER_SIZES[i],
            BUFFER_COUNTS[i]);
    }
}

void* pool_malloc(Pool* pool);
void* pool_calloc(Pool* pool);
void pool_free(Pool* pool, void* mem);

typedef uint32_t bin_index_t;
#define BIN_INDEX_SIZE \
    (ALIGNMENT >= sizeof(bin_index_t) ? ALIGNMENT : sizeof(bin_index_t))

void* hxd_malloc(size_t n) {
    if (n == 0) {
        return NULL;
    }
#ifdef DEBUG_HIST
    printf("POOL:ALLOC:%ld\n", n);
#endif
    n += BIN_INDEX_SIZE;
    size_t bin = align_num(n, BIN_SIZE) / BIN_SIZE;
    Pool* pool = POOLS[bin];
    bin_index_t* ret = pool_malloc(pool);
    if (ret == NULL) {
        return NULL;
    }
    *ret = (bin_index_t)bin;
    return (void*)ret + BIN_INDEX_SIZE;
}

void* hxd_calloc(size_t n, size_t size) {
    void* ret = hxd_malloc(n * size);
    if (ret == NULL) {
        return NULL;
    }
    memset(ret, 0, n);
    return ret;
}

void hxd_free(void* mem) {
    if (mem == NULL) {
        return;
    }
    bin_index_t* orig = (bin_index_t*)(mem - BIN_INDEX_SIZE);
    bin_index_t bin = *orig;
    Pool* pool = POOLS[bin];
    pool_free(pool, orig);
}

Pool* init_pool(void* base, size_t capacity, size_t size, size_t count) {
    Pool* pool = (Pool*)base;
    base += sizeof(Pool);
    size = align_num(size, ALIGNMENT);
    pool->free_list = align_ptr(base);
    pool->size = size;
    struct Chunk* current = pool->free_list;
    for (int i = 0; i < count - 1; i++) {
        struct Chunk* next = (struct Chunk*)((void*)current + size);
        current->next = next;
        current = next;
    }
    current->next = NULL;

    if ((void*)current + size - (void*)pool > capacity) {
        __builtin_trap();
    }
    return pool;
}

void* pool_malloc(Pool* pool) {
    struct Chunk* head = pool->free_list;
    if (head == NULL) {
        return NULL;
    }
    struct Chunk* next = head->next;
    ((Pool*)pool)->free_list = next;
    return (void*)head;
}

void* pool_calloc(Pool* pool) {
    void* ret = pool_malloc(pool);
    if (ret == NULL) {
        return NULL;
    }
    memset(ret, 0, pool->size);
    return ret;
}

void pool_free(Pool* pool, void* mem) {
    struct Chunk* head = pool->free_list;
    ((Pool*)pool)->free_list = (struct Chunk*)mem;
    ((struct Chunk*)mem)->next = head;
}
