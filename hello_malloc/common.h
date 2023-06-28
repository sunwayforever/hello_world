// 2022-08-20 09:52
#ifndef COMMON_H
#define COMMON_H

#ifndef USE_HXD_PREFIX
#define hxd_malloc malloc
#define hxd_calloc calloc
#define hxd_free free
#endif

#include <stddef.h>
#include <string.h>

void init_spaces();
void* hxd_malloc(size_t n);
void* hxd_calloc(size_t n, size_t size);
void hxd_free(void* mem);

#define ALIGNMENT 8

static inline void* align_ptr(void* value) {
    return (
        void*)((((unsigned long int)value + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT);
}

static inline size_t align_num(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

#ifdef DEBUG_HIST
#include "printf.h"
#endif

#endif  // COMMON_H
