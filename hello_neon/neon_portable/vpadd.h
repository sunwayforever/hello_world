// 2023-04-18 11:57
#ifndef VPADD_H
#define VPADD_H

#include <neon_emu_types.h>

int8x8_t vpadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = a.values[i * 2] + a.values[i * 2 + 1];
        r.values[i + 4] = b.values[i * 2] + b.values[i * 2 + 1];
    }
    return r;
}

int16x4_t vpadal_s8(int16x4_t a, int8x8_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] =
            a.values[i] + (int16_t)b.values[i * 2] + b.values[i * 2 + 1];
    }
    return r;
}
#endif  // VPADD_H
