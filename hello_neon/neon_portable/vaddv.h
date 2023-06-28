// 2023-04-18 14:52
#ifndef VADDV_H
#define VADDV_H

#include <neon_emu_types.h>

int8_t vaddv_s8(int8x8_t a) {
    int8_t r = 0;
    for (int i = 0; i < 8; i++) {
        r += a.values[i];
    }
    return r;
}

int16_t vaddlv_s8(int8x8_t a) {
    int16_t r = 0;
    for (int i = 0; i < 8; i++) {
        r += a.values[i];
    }
    return r;
}

#endif  // VADDV_H
