// 2023-04-20 18:49
#ifndef VBIC_H
#define VBIC_H

#include <neon_emu_types.h>
int8x8_t vbic_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] & ~b.values[i];
    }
    return r;
}

#endif  // VBIC_H
