// 2023-04-21 16:22
#ifndef VTRN_H
#define VTRN_H

#include <neon_emu_types.h>

int8x8_t vtrn1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_ilvev_b(b.v.i8, a.v.i8);
    return r;
}

int8x8_t vtrn2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_ilvod_b(b.v.i8, a.v.i8);
    return r;
}
#endif  // VTRN_H
