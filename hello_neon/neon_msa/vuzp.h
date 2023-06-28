// 2023-04-21 16:13
#ifndef VUZP_H
#define VUZP_H

#include <neon_emu_types.h>

int8x8_t vuzp1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int8x16_t _x;
    MERGE(_x, a, b);
    r.v.i8 = __msa_pckev_b(_x.v.i8, _x.v.i8);
    return r;
}
#endif  // VUZP_H
