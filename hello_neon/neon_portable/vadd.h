// 2023-04-14 12:20
#ifndef VADD_H
#define VADD_H

#include <neon_emu_types.h>

int8x8_t vadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i];
    }
    return r;
}

int8x16_t vaddq_s8(int8x16_t a, int8x16_t b) {
    int8x16_t r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] + b.values[i];
    }
    return r;
}

int8x8_t vaddhn_s16(int16x8_t a, int16x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (a.values[i] + b.values[i]) >> 8;
    }
    return r;
}

int8x8_t vhadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (a.values[i] + b.values[i]) >> 1;
    }
    return r;
}

int8x8_t vrhadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (a.values[i] + b.values[i] + 1) >> 1;
    }
    return r;
}

int8x8_t vqadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int16_t tmp = (int16_t)a.values[i] + b.values[i];
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        }
        if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = (int8_t)tmp;
    }
    return r;
}

int8x8_t vuqadd_s8(int8x8_t a, uint8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int16_t tmp = (int16_t)a.values[i] + b.values[i];
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        }
        if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = (int8_t)tmp;
    }
    return r;
}

uint8x8_t vsqadd_u8(uint8x8_t a, int8x8_t b) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        int16_t tmp = (int16_t)a.values[i] + b.values[i];
        if (tmp > UINT8_MAX) {
            tmp = UINT8_MAX;
        }
        if (tmp < 0) {
            tmp = 0;
        }
        r.values[i] = (uint8_t)tmp;
    }
    return r;
}

int8_t vqaddb_s8(int8_t a, int8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > INT8_MAX) {
        r = INT8_MAX;
    }
    if (r < INT8_MIN) {
        r = INT8_MIN;
    }
    return (int8_t)r;
}

int8_t vuqaddb_s8(int8_t a, uint8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > INT8_MAX) {
        r = INT8_MAX;
    }
    if (r < INT8_MIN) {
        r = INT8_MIN;
    }
    return (int8_t)r;
}

uint8_t vsqaddb_u8(uint8_t a, int8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > UINT8_MAX) {
        r = UINT8_MAX;
    }
    if (r < 0) {
        r = 0;
    }
    return (uint8_t)r;
}

int16x8_t vaddl_s8(int8x8_t a, int8x8_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (int16_t)a.values[i] + b.values[i];
    }
    return r;
}

int16x8_t vaddl_high_s8(int8x16_t a, int8x16_t b) {
    int16x8_t r;
    // a: abcdefg
    // b: abcdefg
    for (int i = 0; i < 8; i++) {
        r.values[i] = (int16_t)a.values[i + 8] + b.values[i + 8];
    }
    return r;
}

int16x8_t vaddw_s8(int16x8_t a, int8x8_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i];
    }
    return r;
}

int16x8_t vaddw_high_s8(int16x8_t a, int8x16_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i + 8];
    }
    return r;
}

#endif  // VADD_H
