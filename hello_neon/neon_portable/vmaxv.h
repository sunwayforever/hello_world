// 2023-04-18 14:52
#ifndef VMAXV_H
#define VMAXV_H

#include <math.h>

#include <neon_emu_types.h>

float vmaxv_f32(float32x2_t a) {
    float r = -INFINITY;
    for (int i = 0; i < 2; i++) {
        if (r < a.values[i]) {
            r = a.values[i];
        }
    }
    return r;
}

#endif  // VMAXV_H
