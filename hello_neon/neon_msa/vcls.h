// 2023-04-20 17:45
#ifndef VCLS_H
#define VCLS_H

#include <neon_emu_types.h>
int8x8_t vcls_s8(int8x8_t a) {
    int8x8_t r = {0};
    int8x8_t one, zero;
    one.v.i8 = __msa_nloc_b(a.v.i8);
    zero.v.i8 = __msa_nlzc_b(a.v.i8);
    r.v.i8 = __msa_max_s_b(zero.v.i8, one.v.i8);
    r.v.i8 = __msa_subvi_b(r.v.i8, 1);
    return r;
}

int8x8_t vcls_u8(uint8x8_t a) { return vcls_s8(vreinterpret_s8_u8(a)); }

#endif  // VCLS_H
