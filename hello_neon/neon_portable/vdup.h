// 2023-04-21 13:43
#ifndef VDUP_H
#define VDUP_H

#include <neon_emu_types.h>

int8x8_t vdup_n_s8(int8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a;
    }
    return r;
}
#endif  // VDUP_H
