// 2023-04-21 14:07
#ifndef VGET_H
#define VGET_H

#include <neon_emu_types.h>

int8x8_t vget_high_s8(int8x16_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i + 8];
    }
    return r;
}

int8_t vget_lane_s8(int8x8_t a, int n) { return a.values[n]; }
#endif  // VGET_H
