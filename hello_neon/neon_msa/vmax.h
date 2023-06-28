// 2023-04-17 17:50
#ifndef VMAX_H
#define VMAX_H

#include <math.h>
#include <neon_emu_types.h>

float32x2_t vmax_f32(float32x2_t a, float32x2_t b) {
    float32x2_t r;
    r.v.f32 = __msa_fmax_w(a.v.f32, b.v.f32);
    return r;
}

// NOTE: msa 不支持 nm
float32x2_t vmaxnm_f32(float32x2_t a, float32x2_t b) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = fmax(a.values[i], b.values[i]);
    }
    return r;
}
#endif  // VMAX_H
