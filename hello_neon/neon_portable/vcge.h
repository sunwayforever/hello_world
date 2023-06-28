// 2023-04-18 17:01
#ifndef VCGE_H
#define VCGE_H

#include <neon_emu_types.h>

uint8x8_t vcgez_s8(int8x8_t a) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (a.values[i] >= 0) {
            r.values[i] = UINT8_MAX;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}
#endif  // VCGE_H
