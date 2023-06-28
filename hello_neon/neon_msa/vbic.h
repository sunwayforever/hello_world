// 2023-04-20 18:49
#ifndef VBIC_H
#define VBIC_H

#include <neon_emu_types.h>
int8x8_t vbic_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r = {0};
    int8x8_t tmp;
    tmp.v.i8 = __msa_fill_b(0xff);
    tmp.v.u8 = __msa_xor_v(b.v.u8, tmp.v.u8);
    r.v.u8 = __msa_and_v(a.v.u8, tmp.v.u8);
    return r;
}

#endif  // VBIC_H
