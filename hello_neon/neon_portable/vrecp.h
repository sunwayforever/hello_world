// 2023-04-18 10:47
#ifndef VRECP_H
#define VRECP_H

#include <math.h>

#include <neon_emu_types.h>

float32x2_t vrecpe_f32(float32x2_t a) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = 1.0f / a.values[i];
    }
    return r;
}

float32x2_t vrecps_f32(float32x2_t a, float32x2_t b) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = 2.0 - a.values[i] * b.values[i];
    }
    return r;
}

float vrsqrtes_f32(float a) { return sqrtf(1.0f / a); }

#endif  // VRECP_H
