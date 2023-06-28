// 2023-04-17 18:36
#ifndef VRND_H
#define VRND_H

#include <fenv.h>
#include <math.h>
#include <neon_emu_types.h>

typedef enum {
    DEFAULT = 0,
    ZERO = 1,
    PLUS = 2,
    MINUS = 3,
} RM;

static void set_rounding_mode(RM m) {
    int32_t msacsr = __msa_cfcmsa(1);
    msacsr = (msacsr & 0xffffffffc) | m;
    // NOTE: it's weird that __builtin_msa_ctcmsa is not wrapped to __msa_ctcmsa
    __builtin_msa_ctcmsa(1, msacsr);
}

float32x2_t vrnd_f32(float32x2_t a) {
    float32x2_t r;
    set_rounding_mode(ZERO);
    r.v.f32 = __msa_frint_w(a.v.f32);
    set_rounding_mode(DEFAULT);
    return r;
}

float32x2_t vrndn_f32(float32x2_t a) {
    float32x2_t r;
    set_rounding_mode(DEFAULT);
    r.v.f32 = __msa_frint_w(a.v.f32);
    return r;
}

float32x2_t vrndm_f32(float32x2_t a) {
    float32x2_t r;
    set_rounding_mode(MINUS);
    r.v.f32 = __msa_frint_w(a.v.f32);
    set_rounding_mode(DEFAULT);
    return r;
}
#endif  // VRND_H
