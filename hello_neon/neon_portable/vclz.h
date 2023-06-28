// 2023-04-20 17:58
#ifndef VCLZ_H
#define VCLZ_H

#include <neon_emu_types.h>

int8x8_t vclz_s8(int8x8_t a) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        int8_t tmp = a.values[i];
        for (int j = 0; j < 8; j++) {
            if ((tmp & (1 << 7)) != 0) {
                break;
            }
            r.values[i] += 1;
            tmp <<= 1;
        }
    }
    return r;
}

#endif  // VCLZ_H
