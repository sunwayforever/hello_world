// 2023-04-20 18:15
#ifndef VCNT_H
#define VCNT_H

#include <neon_emu_types.h>

int8x8_t vcnt_s8(int8x8_t a) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        uint8_t tmp = a.values[i];
        for (int j = 0; j < 8; j++) {
            if ((tmp & 1) == 1) {
                r.values[i] += 1;
            }
            tmp >>= 1;
        }
    }
    return r;
}

#endif  // VCNT_H
