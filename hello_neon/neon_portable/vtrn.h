// 2023-04-21 16:22
#ifndef VTRN_H
#define VTRN_H

#include <neon_emu_types.h>

int8x8_t vtrn1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[2 * i] = a.values[2 * i];
        r.values[2 * i + 1] = b.values[2 * i];
    }
    return r;
}

int8x8_t vtrn2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[2 * i] = a.values[2 * i + 1];
        r.values[2 * i + 1] = b.values[2 * i + 1];
    }
    return r;
}

#endif  // VTRN_H
