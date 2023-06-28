// 2023-04-18 10:47
#ifndef VRECP_H
#define VRECP_H

#include <math.h>
#include <neon_emu_types.h>

float32x2_t vrecpe_f32(float32x2_t a) {
    float32x2_t r;
    r.v.f32 = __msa_frcp_w(a.v.f32);
    return r;
}

float32x2_t vrecps_f32(float32x2_t a, float32x2_t b) {
    float32x2_t r;
    float32x2_t two;
    two.values[0] = 2.0;
    two.values[1] = 2.0;
    r.v.f32 = __msa_fsub_w(two.v.f32, __msa_fmul_w(a.v.f32, b.v.f32));
    return r;
}

float vrsqrtes_f32(float a) { return sqrtf(1.0f / a); }

#endif  // VRECP_H
