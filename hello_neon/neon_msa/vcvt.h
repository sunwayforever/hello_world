// 2023-04-20 13:58
#ifndef VCVT_H
#define VCVT_H
#include <fenv.h>
#include <math.h>
#include <neon_emu_types.h>

int32x2_t vcvt_s32_f32(float32x2_t a) {
    int32x2_t r;
    r.v.i32 = __msa_ftrunc_s_w(a.v.f32);
    return r;
}

int32x2_t vcvta_s32_f32(float32x2_t a) {
    int32x2_t r;
    r.v.i32 = __msa_ftint_s_w(a.v.f32);
    return r;
}

#endif  // VCVT_H
