// 2023-04-28 15:22
#ifndef FLOAT_CONVERT_H
#define FLOAT_CONVERT_H

#include <math.h>
#include <stdint.h>

#include "msa_emu_types.h"

v2f64 __msa_fexupr_d(v4f32 a) {
    v2f64 r;
    for (int i = 0; i < 2; i++) {
        r.values[i] = a.values[i];
    }
    return r;
}

v4u32 __msa_ftint_u_w(v4f32 a) {
    v4u32 r;
    for (int i = 0; i < 4; i++) {
        float32_t tmp = a.values[i];
        if (isnan(tmp) || tmp < 0.0f) {
            r.values[i] = 0;
        } else {
            r.values[i] = nearbyintf(a.values[i]);
        }
    }
    return r;
}

v4i32 __msa_ftrunc_s_w(v4f32 a) {
    v4i32 r;
    for (int i = 0; i < 4; i++) {
        float32_t tmp = a.values[i];
        if (isnan(tmp)) {
            r.values[i] = 0;
        } else {
            r.values[i] = (int32_t)a.values[i];
        }
    }
    return r;
}

v8i16 __msa_ftq_h(v4f32 a, v4f32 b) {
    v8i16 r;
    for (int i = 0; i < 4; i++) {
        float32_t _b = nearbyintf(b.values[i] * (1 << 15));
        float32_t _a = nearbyintf(a.values[i] * (1 << 15));
        if (_b > 32767) _b = 32767;
        if (_a > 32767) _a = 32767;
        if (_b < -32768) _b = -32768;
        if (_a < -32768) _a = -32768;
        r.values[i] = _b;
        r.values[i + 4] = _a;
    }
    return r;
}

#endif  // FLOAT_CONVERT_H
