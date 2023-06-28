// 2023-04-21 15:34
#ifndef VREV_H
#define VREV_H

#include <neon_emu_types.h>

int8x8_t vrev16_s8(int8x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 4; i++) {
        r.values[i * 2] = a.values[i * 2 + 1];
        r.values[i * 2 + 1] = a.values[i * 2];
    }
    return r;
}

#endif  // VREV_H
