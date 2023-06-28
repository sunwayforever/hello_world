// 2023-04-18 14:24
#ifndef VPMAX_H
#define VPMAX_H

#include <neon_emu_types.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

int8x8_t vpmax_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = MAX(a.values[2 * i], a.values[2 * i + 1]);
        r.values[i + 4] = MAX(b.values[2 * i], b.values[2 * i + 1]);
    }
    return r;
}

#endif  // VPMAX_H
