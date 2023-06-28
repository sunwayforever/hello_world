// 2023-04-19 16:24
#ifndef VSHR_H
#define VSHR_H

#include <stdint.h>

#include <neon_emu_types.h>

int16x4_t vshr_n_s16(int16x4_t a, int n) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = a.values[i] >> n;
    }
    return r;
}

uint16x4_t vshr_n_u16(uint16x4_t a, int n) {
    uint16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = a.values[i] >> n;
    }
    return r;
}

int8x8_t vrshr_n_s8(int8x8_t a, int n) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (a.values[i] + (1 << (n - 1))) >> n;
    }
    return r;
}

int8x8_t vsra_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + (b.values[i] >> n);
    }
    return r;
}

int8x8_t vrsra_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + ((b.values[i] + (1 << (n - 1))) >> n);
    }
    return r;
}

int8x8_t vshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] >> n;
    }
    return r;
}

int8x16_t vshrn_high_n_s16(int8x8_t a, int16x8_t b, int n) {
    int8x16_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i];
        r.values[i + 8] = b.values[i] >> n;
    }
    return r;
}

uint8x8_t vqshrun_n_s16(int16x8_t a, int n) {
    uint8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = a.values[i] >> n;
        if (tmp > UINT8_MAX) {
            tmp = UINT8_MAX;
        }
        if (tmp < 0) {
            tmp = 0;
        }
        r.values[i] = tmp;
    }
    return r;
}

int8x8_t vqshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = a.values[i] >> n;
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        }
        if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = tmp;
    }
    return r;
}

int8x8_t vqrshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = (a.values[i] + (1 << (n - 1))) >> n;
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        }
        if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = tmp;
    }
    return r;
}

int8x8_t vrshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (a.values[i] + (1 << (n - 1))) >> n;
    }
    return r;
}

int8x8_t vsri_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    uint8_t mask = (1 << (8 - n)) - 1;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] & ~mask | ((b.values[i] >> n) & mask);
    }
    return r;
};
#endif  // VSHR_H
