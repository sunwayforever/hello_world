// 2023-04-18 18:15
#ifndef VTST_H
#define VTST_H

#include <neon_emu_types.h>
#include <stdint.h>

uint8x8_t vtst_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    uint8x8_t tmp;
    int8x8_t zero;

    zero.v.i8 = __msa_fill_b(0);
    tmp.v.u8 = __msa_and_v(a.v.u8, b.v.u8);
    r.v.i8 = __msa_clt_u_b(zero.v.u8, tmp.v.u8);
    return r;
}
#endif  // VTST_H
