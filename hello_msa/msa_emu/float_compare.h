// 2023-04-28 14:16
#ifndef FLOAT_COMPARE_H
#define FLOAT_COMPARE_H
#include <math.h>

#include "msa_emu_types.h"

v4i32 __msa_fcun_w(v4f32 a, v4f32 b) {
    v4i32 r;
    for (int i = 0; i < 4; i++) {
        if (isnan(a.values[i]) || isnan(b.values[i])) {
            r.values[i] = -1;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}

v4i32 __msa_fcult_w(v4f32 a, v4f32 b) {
    v4i32 r;
    for (int i = 0; i < 4; i++) {
        if (isnan(a.values[i]) || isnan(b.values[i]) ||
            a.values[i] < b.values[i]) {
            r.values[i] = -1;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}

#endif  // FLOAT_COMPARE_H
