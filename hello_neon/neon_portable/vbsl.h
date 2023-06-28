// 2023-04-20 18:57
#ifndef VBSL_H
#define VBSL_H

#include <stdint.h>

#include <neon_emu_types.h>
int8x8_t vbsl_s8(uint8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        uint8_t mask = 1;
        for (int j = 0; j < 8; j++) {
            if (a.values[i] & mask) {
                r.values[i] |= b.values[i] & mask;
            } else {
                r.values[i] |= c.values[i] & mask;
            }
            mask <<= 1;
        }
    }
    return r;
}

#endif  // VBSL_H
