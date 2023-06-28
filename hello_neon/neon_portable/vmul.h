// 2023-04-14 19:39
#ifndef VMUL_H
#define VMUL_H

#include <neon_emu_types.h>

int8x8_t vmul_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] * b.values[i];
    }
    return r;
}

int8x16_t vmulq_s8(int8x16_t a, int8x16_t b) {
    int8x16_t r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] * b.values[i];
    }
    return r;
}

int8x8_t vmla_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i] * c.values[i];
    }
    return r;
}

int8x8_t vmls_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] - b.values[i] * c.values[i];
    }
    return r;
}

int16x8_t vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i] * c.values[i];
    }
    return r;
}

int16x8_t vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] - b.values[i] * c.values[i];
    }
    return r;
}

int16x8_t vmlal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] + b.values[i + 8] * c.values[i + 8];
    }
    return r;
}

int16x8_t vmlsl_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] - b.values[i + 8] * c.values[i + 8];
    }
    return r;
}

float32x2_t vfma_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = (double)a.values[i] + (double)b.values[i] * c.values[i];
    }
    return r;
}

float32x2_t vfma_lane_f32(
    float32x2_t a, float32x2_t b, float32x2_t v, int lane) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] =
            (double)a.values[i] + (double)b.values[i] * v.values[lane];
    }
    return r;
}

float32x2_t vfma_laneq_f32(
    float32x2_t a, float32x2_t b, float32x4_t v, int lane) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] =
            (double)a.values[i] + (double)b.values[i] * v.values[lane];
    }
    return r;
}

float vfmas_lane_f32(float a, float b, float32x2_t v, int lane) {
    float r = (double)a + (double)b * v.values[lane];
    return r;
}

int16x4_t vqdmulh_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = (a.values[i] * b.values[i] * 2) >> 16;
    }
    return r;
}

int16x4_t vqrdmulh_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = (a.values[i] * b.values[i] * 2 + (1 << 15)) >> 16;
    }
    return r;
}

int32x4_t vqdmull_s16(int16x4_t a, int16x4_t b) {
    int32x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = (a.values[i] * b.values[i] * 2);
    }
    return r;
}

int16x8_t vmull_s8(int8x8_t a, int8x8_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (int16_t)a.values[i] * b.values[i];
    }
    return r;
}

int16x8_t vmull_high_s8(int8x16_t a, int8x16_t b) {
    int16x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = (int16_t)a.values[i + 8] * b.values[i + 8];
    }
    return r;
}

int32x4_t vmull_n_s16(int16x4_t a, int16_t b) {
    int32x4_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = (int32_t)a.values[i] * b;
    }
    return r;
};

poly8x8_t vmul_p8(poly8x8_t a, poly8x8_t b) {
    poly8x8_t r;
    for (int i = 0; i < 8; i++) {
        uint16_t tmp = 0;
        uint8_t x = a.values[i];
        uint8_t y = b.values[i];
        for (int j = 0; j < 8; j++) {
            // x=1111
            // y=1010
            if (y & 1) {
                tmp ^= x;
            }
            y >>= 1;
            x <<= 1;
        }
        r.values[i] = (uint8_t)tmp;
    }
    return r;
};
#endif  // VMUL_H
