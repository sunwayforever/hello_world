// 2023-04-20 14:53
#ifndef VMOV_H
#define VMOV_H

#include <stdint.h>

#include <neon_emu_types.h>

int8x8_t vmovn_s16(int16x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i];
    }
    return r;
}

int8x16_t vmovn_high_s16(int8x8_t r, int16x8_t a) {
    int8x16_t res;
    for (int i = 0; i < 8; i++) {
        res.values[i] = r.values[i];
        res.values[i + 8] = a.values[i];
    }
    return res;
}

int16x8_t vmovl_s8(int8x8_t a) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i];
    }
    return r;
}

int16x8_t vmovl_high_s8(int8x16_t a) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i + 8];
    }
    return r;
}

int8x8_t vqmovn_s16(int16x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (a.values[i] > INT8_MAX) {
            r.values[i] = INT8_MAX;
        } else if (a.values[i] < INT8_MIN) {
            r.values[i] = INT8_MIN;
        } else {
            r.values[i] = a.values[i];
        }
    }
    return r;
}

uint8x8_t vqmovun_s16(int16x8_t a) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (a.values[i] > UINT8_MAX) {
            r.values[i] = UINT8_MAX;
        } else if (a.values[i] < 0) {
            r.values[i] = 0;
        } else {
            r.values[i] = a.values[i];
        }
    }
    return r;
}
#endif  // VMOV_H
