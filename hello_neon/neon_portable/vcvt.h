// 2023-04-20 13:58
#ifndef VCVT_H
#define VCVT_H
#include <fenv.h>
#include <math.h>
#include <neon_emu_types.h>

int32x2_t vcvt_s32_f32(float32x2_t a) {
    int32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = a.values[i];
    }
    return r;
}

int32x2_t vcvta_s32_f32(float32x2_t a) {
    int32x2_t r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = nearbyintf(a.values[i]);
    }
    return r;
}

#endif  // VCVT_H
