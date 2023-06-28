// 2023-04-18 11:38
#ifndef VSQRT_H
#define VSQRT_H

#include <math.h>

#include <neon_emu_types.h>

float32x2_t vsqrt_f32(float32x2_t a) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = sqrtf(a.values[i]);
    }
    return r;
}
#endif  // VSQRT_H
