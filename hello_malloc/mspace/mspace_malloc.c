#include "mspace_malloc.h"

#include <stdint.h>

#include "dlmalloc.h"

void* init_mspace(void* base, size_t capacity) {
    return create_mspace_with_base(base, capacity, 0);
}

extern int N_SPACES;
extern void* SPACES[];
extern size_t SPACE_SIZES[];

void init_spaces() {
    for (int i = 0; i < N_SPACES; i++) {
        SPACES[i] = init_mspace(SPACES[i], SPACE_SIZES[i]);
    }
}

typedef uint32_t space_index_t;
#define SPACE_INDEX_SIZE \
    (ALIGNMENT >= sizeof(space_index_t) ? ALIGNMENT : sizeof(space_index_t))

static void* internal_mspace_malloc(size_t n) {
    if (N_SPACES == 1) {
        return mspace_malloc(SPACES[0], n);
    }
    for (int i = 0; i < N_SPACES; i++) {
        space_index_t* ret = mspace_malloc(SPACES[i], n + SPACE_INDEX_SIZE);
        if (ret == NULL) {
            continue;
        }
        *ret = i;
        return (void*)ret + SPACE_INDEX_SIZE;
    }
    return NULL;
}

static void interal_mspace_free(void* mem) {
    if (N_SPACES == 1) {
        mspace_free(SPACES[0], mem);
        return;
    }
    space_index_t* orig = (space_index_t*)(mem - SPACE_INDEX_SIZE);
    void* space = SPACES[*orig];
    mspace_free(space, (void*)orig);
}

void* hxd_malloc(size_t n) {
    if (n == 0) {
        return NULL;
    }
#ifdef DEBUG_HIST
    // NOTE: pool_malloc has an `int` header for every malloc, on both m32 and
    // m64, it must align with `BIN_INDEX_SIZE` in pool_malloc.c
    printf("MSPACE:ALLOC:%ld\n", ALIGNMENT >= 4 ? n + ALIGNMENT : n + 4);
    size_t* ret = (size_t*)internal_mspace_malloc(n + sizeof(size_t));
    *ret = n;
    return (void*)(ret + 1);
#else
    return internal_mspace_malloc(n);
#endif
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
#ifdef DEBUG_HIST
    size_t* orig = (size_t*)mem - 1;
    size_t n = *orig;
    printf("MSPACE:DEALLOC:%ld\n", ALIGNMENT >= 4 ? n + ALIGNMENT : n + 4);
    interal_mspace_free((void*)orig);
#else
    interal_mspace_free(mem);
#endif
}
