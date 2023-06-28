// 2023-04-17 15:40
#ifndef VABD_H
#define VABD_H

#include <neon_emu_types.h>

int8x8_t vabd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = (int16_t)a.values[i] - b.values[i];
        tmp = tmp > 0 ? tmp : -tmp;
        r.values[i] = (int8_t)tmp;
    }
    return r;
}

int8x8_t vaba_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = (int16_t)b.values[i] - c.values[i];
        tmp = tmp > 0 ? tmp : -tmp;
        r.values[i] = a.values[i] + (int8_t)tmp;
    }
    return r;
}

int8x8_t vabs_s8(int8x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] > 0 ? a.values[i] : -a.values[i];
    }
    return r;
}

int8_t vqabsb_s8(int8_t a) {
    int16_t tmp;
    tmp = a > 0 ? a : -a;
    if (tmp > INT8_MAX) {
        tmp = INT8_MAX;
    }
    return (int8_t)tmp;
}

#endif  // VABD_H
