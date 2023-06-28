// 2023-04-21 15:58
#ifndef VZIP_H
#define VZIP_H

#include <neon_emu_types.h>

int8x8_t vzip1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i * 2] = a.values[i];
        r.values[i * 2 + 1] = b.values[i];
    }
    return r;
}

int8x8_t vzip2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i * 2] = a.values[i + 4];
        r.values[i * 2 + 1] = b.values[i + 4];
    }
    return r;
}

int8x8x2_t vzip_s8(int8x8_t a, int8x8_t b) {
    int8x8x2_t r;
    for (int i = 0; i < 4; i++) {
        r.val[0].values[i * 2] = a.values[i];
        r.val[0].values[i * 2 + 1] = b.values[i];
        r.val[1].values[i * 2] = a.values[i + 4];
        r.val[1].values[i * 2 + 1] = b.values[i + 4];
    }
    return r;
}

#endif  // VZIP_H
