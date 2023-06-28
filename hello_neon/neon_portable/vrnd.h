// 2023-04-17 18:36
#ifndef VRND_H
#define VRND_H

#include <fenv.h>
#include <math.h>

#include <neon_emu_types.h>

float32x2_t vrnd_f32(float32x2_t a) {
    float32x2_t r;
    fesetround(FE_TOWARDZERO);
    for (int i = 0; i < 2; i++) {
        // r.values[i] =
        //     a.values[i] > 0.0f ? floorf(a.values[i]) : ceilf(a.values[i]);
        r.values[i] = nearbyintf(a.values[i]);
    }
    return r;
}

float32x2_t vrndn_f32(float32x2_t a) {
    float32x2_t r;
    fesetround(FE_TONEAREST);
    for (int i = 0; i < 2; i++) {
        r.values[i] = rint(a.values[i]);
    }
    return r;
}

float32x2_t vrndm_f32(float32x2_t a) {
    float32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = floorf(a.values[i]);
    }
    return r;
}
#endif  // VRND_H
