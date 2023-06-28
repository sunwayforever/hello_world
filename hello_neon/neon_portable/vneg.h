// 2023-04-20 16:45
#ifndef VNEG_H
#define VNEG_H

#include <neon_emu_types.h>

int8x16_t vnegq_s8(int8x16_t a) {
    int8x16_t r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = -a.values[i];
    }
    return r;
}

int8x8_t vqneg_s8(int8x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (a.values[i] == INT8_MIN) {
            r.values[i] = INT8_MAX;
        } else {
            r.values[i] = -a.values[i];
        }
    }
    return r;
}

int8_t vqnegb_s8(int8_t a) {
    if (a == INT8_MIN) {
        return INT8_MAX;
    }
    return -a;
}
#endif  // VNEG_H
