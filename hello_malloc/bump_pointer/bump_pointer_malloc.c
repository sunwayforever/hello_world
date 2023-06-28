#include "bump_pointer_malloc.h"

extern int N_SPACES;
extern void* SPACES[];
extern size_t SPACE_SIZES[];

typedef struct {
    size_t buffer_size;
    size_t current_offset;
    void* buffer;
} MSpace;

static void* init_mspace(void* base, size_t capacity) {
    MSpace* space = (MSpace*)base;
    space->current_offset = 0;
    space->buffer = align_ptr(base + sizeof(MSpace));
    space->buffer_size = base + capacity - space->buffer;
    return space;
}

void init_spaces() {
    for (int i = 0; i < N_SPACES; i++) {
        SPACES[i] = init_mspace(SPACES[i], SPACE_SIZES[i]);
    }
}

void* mspace_malloc(MSpace* space, size_t n) {
    if (space->current_offset + n > space->buffer_size) {
        return NULL;
    }
#ifdef DEBUG_HIST
    printf("BUMP:ALLOC:%ld\n", n);
#endif
    void* ret = space->buffer + space->current_offset;
    space->current_offset = align_num(space->current_offset + n, ALIGNMENT);
    return ret;
}

void* hxd_malloc(size_t n) {
    if (N_SPACES == 1) {
        return mspace_malloc(SPACES[0], n);
    }
    for (int i = 0; i < N_SPACES; i++) {
        void* ret = mspace_malloc(SPACES[i], n);
        if (ret == NULL) {
            continue;
        }
        return ret;
    }
    return NULL;
}

void* hxd_calloc(size_t n, size_t size) {
    void* ret = hxd_malloc(n * size);
    if (ret == NULL) {
        return NULL;
    }
    memset(ret, 0, n);
    return ret;
}

void free(void* mem) {}
