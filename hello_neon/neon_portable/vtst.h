// 2023-04-18 18:15
#ifndef VTST_H
#define VTST_H

#include <stdint.h>

#include <neon_emu_types.h>

uint8x8_t vtst_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        if ((a.values[i] & b.values[i]) != 0) {
            r.values[i] = UINT8_MAX;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}
#endif  // VTST_H
