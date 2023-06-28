// 2023-04-21 11:27
#ifndef VCREATE_H
#define VCREATE_H

#include <neon_emu_types.h>

int8x8_t vcreate_s8(uint64_t a) {
    int8x8_t r;
    memcpy(&r, &a, sizeof(a));
    return r;
}

#endif  // VCREATE_H
