// 2023-04-18 17:01
#ifndef VCGE_H
#define VCGE_H

#include <neon_emu_types.h>

uint8x8_t vcgez_s8(int8x8_t a) {
    uint8x8_t r;
    int8x8_t zero;
    zero.v.i8 = __msa_fill_b(0);
    r.v.i8 = __msa_cle_s_b(zero.v.i8, a.v.i8);
    return r;
}
#endif  // VCGE_H
