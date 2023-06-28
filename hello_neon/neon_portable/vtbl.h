// 2023-04-21 17:35
#ifndef VTBL_H
#define VTBL_H

#include <stdint.h>

#include <neon_emu_types.h>

int8x8_t vtbl1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t index = b.values[i];
        r.values[i] = index >= 0 && index <= 7 ? a.values[index] : 0;
    }
    return r;
}

int8x8_t vtbl2_s8(int8x8x2_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t index = b.values[i];
        if (index >= 0 && index <= 7) {
            r.values[i] = a.val[0].values[index];
        } else if (index >= 8 && index <= 15) {
            r.values[i] = a.val[1].values[index - 8];
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}

// NOTE: 与 vtbl2 类似, 唯一不同的是 index 无效时使用 a 的值 (而不是赋为 0)
int8x8_t vtbx1_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t index = c.values[i];
        if (index >= 0 && index <= 7) {
            r.values[i] = b.values[index];
        } else if (index >= 8 && index <= 15) {
            r.values[i] = a.values[index - 8];
        } else {
            r.values[i] = a.values[i];
        }
    }
    return r;
}

int8x8_t vtbx2_s8(int8x8_t a, int8x8x2_t b, int8x8_t c) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t index = c.values[i];
        if (index >= 0 && index <= 7) {
            r.values[i] = b.val[0].values[index];
        } else if (index >= 8 && index <= 15) {
            r.values[i] = b.val[1].values[index - 8];
        } else if (index >= 16 && index <= 23) {
            r.values[i] = a.values[index - 8];
        } else {
            r.values[i] = a.values[i];
        }
    }
    return r;
}

#endif  // VTBL_H
