// 2023-04-21 16:13
#ifndef VUZP_H
#define VUZP_H

#include <neon_emu_types.h>

int8x8_t vuzp1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i] = a.values[i * 2];
        r.values[i + 4] = b.values[i * 2];
    }
    return r;
}
#endif  // VUZP_H
