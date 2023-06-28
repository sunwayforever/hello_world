// 2023-04-18 11:38
#ifndef VSQRT_H
#define VSQRT_H

#include <neon_emu_types.h>

float32x2_t vsqrt_f32(float32x2_t a) {
    float32x2_t r;
    r.v.f32 = __msa_fsqrt_w(a.v.f32);
    return r;
}
#endif  // VSQRT_H
