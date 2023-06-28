// 2023-04-18 17:01
#ifndef VCGT_H
#define VCGT_H

#include <neon_emu_types.h>

uint8x8_t vcgt_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (a.values[i] > b.values[i]) {
            r.values[i] = UINT8_MAX;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}
#endif  // VCGT_H
