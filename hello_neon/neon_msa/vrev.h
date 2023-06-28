// 2023-04-21 15:34
#ifndef VREV_H
#define VREV_H

#include <neon_emu_types.h>

int8x8_t vrev16_s8(int8x8_t a) {
    int8x8_t r;
    int8x8_t shuffle;
    // 1 0 3 2 5 4
    // 0   1   2
    for (int i = 0; i < 4; i++) {
        shuffle.values[i * 2] = i * 2 + 1;
        shuffle.values[i * 2 + 1] = i * 2;
    }
    r.v.i8 = __msa_vshf_b(shuffle.v.i8, a.v.i8, a.v.i8);
    return r;
}

#endif  // VREV_H
