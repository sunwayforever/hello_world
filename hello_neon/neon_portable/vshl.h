// 2023-04-19 10:45
#ifndef VSHL_H
#define VSHL_H

#include <stdint.h>

#include <neon_emu_types.h>

int16x4_t vshl_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        int8_t shift = b.values[i] & 0xff;
        if (shift > 16) {
            shift = 16;
        }
        if (shift < -16) {
            shift = -16;
        }
        if (shift < 0) {
            r.values[i] = a.values[i] >> -shift;
        } else {
            r.values[i] = a.values[i] << shift;
        }
    }
    return r;
}

uint8x8_t vshl_u8(uint8x8_t a, int8x8_t b) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t shift = b.values[i];
        if (shift > 8) {
            shift = 8;
        }
        if (shift < -8) {
            shift = -8;
        }
        if (shift < 0) {
            r.values[i] = a.values[i] >> -shift;
        } else {
            r.values[i] = a.values[i] << shift;
        }
    }
    return r;
}

int16x8_t vshlq_s16(int16x8_t a, int16x8_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t shift = b.values[i] & 0xff;
        if (shift > 16) {
            shift = 16;
        }
        if (shift < -16) {
            shift = -16;
        }
        if (shift < 0) {
            r.values[i] = a.values[i] >> -shift;
        } else {
            r.values[i] = a.values[i] << shift;
        }
    }
    return r;
}

int16x4_t vshl_n_s16(int16x4_t a, int16_t n) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        int8_t shift = n & 0xff;
        if (shift > 16) {
            shift = 16;
        }
        if (shift < -16) {
            shift = -16;
        }
        if (shift < 0) {
            r.values[i] = a.values[i] >> -shift;
        } else {
            r.values[i] = a.values[i] << shift;
        }
    }
    return r;
}

int16x4_t vqshl_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        int8_t shift = b.values[i] & 0xff;
        if (shift > 16) {
            shift = 16;
        }
        if (shift < -16) {
            shift = -16;
        }
        int32_t tmp;
        if (shift < 0) {
            tmp = a.values[i] >> -shift;
        } else {
            tmp = a.values[i] << shift;
        }
        if (tmp > INT16_MAX) {
            tmp = INT16_MAX;
        }
        if (tmp < INT16_MIN) {
            tmp = INT16_MIN;
        }
        r.values[i] = tmp;
    }
    return r;
}

uint64x1_t vqshl_u64(uint64x1_t a, int64x1_t b) {
    uint64x1_t r;
    for (int i = 0; i < 1; i++) {
        int8_t shift = b.values[i] & 0xff;
        if (shift > 64) {
            shift = 64;
        }
        if (shift < -64) {
            shift = -64;
        }
        uint64_t tmp, prev_tmp;
        if (shift < 0) {
            tmp = a.values[i] >> -shift;
        } else {
            tmp = a.values[i];
            for (int i = 0; i < shift; i++) {
                prev_tmp = tmp;
                tmp <<= 1;
                if (tmp < prev_tmp) {
                    tmp = UINT64_MAX;
                    break;
                }
            }
        }
        r.values[i] = tmp;
    }
    return r;
}

int16x4_t vrshl_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        int8_t shift = b.values[i] & 0xff;
        if (shift > 16) {
            shift = 16;
        }
        if (shift < -16) {
            shift = -16;
        }
        if (shift < 0) {
            r.values[i] = (a.values[i] + (1 << (-shift - 1))) >> -shift;
        } else {
            r.values[i] = a.values[i] << shift;
        }
    }
    return r;
}

int16x8_t vshll_n_s8(int8x8_t a, int n) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t shift = n & 0xff;
        if (shift > 8) {
            shift = 8;
        }
        if (shift < -8) {
            shift = -8;
        }
        if (shift < 0) {
            r.values[i] = (int16_t)a.values[i] >> -shift;
        } else {
            r.values[i] = (int16_t)a.values[i] << shift;
        }
    }
    return r;
};

int16x4_t vsli_n_s16(int16x4_t a, int16x4_t b, int n) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = (a.values[i] & ((1 << n) - 1)) + (b.values[i] << n);
    }
    return r;
}
#endif  // VSHL_H
