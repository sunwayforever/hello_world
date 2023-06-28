// 2023-04-18 17:46
#ifndef VCAGE_H
#define VCAGE_H

#include <math.h>

#include <neon_emu_types.h>

uint32x2_t vcage_f32(float32x2_t a, float32x2_t b) {
    uint32x2_t r;
    for (int i = 0; i < 2; i++) {
        if (fabs(a.values[i]) >= fabs(b.values[i])) {
            r.values[i] = UINT32_MAX;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}

#endif  // VCAGE_H
