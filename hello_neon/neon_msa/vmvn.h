// 2023-04-20 17:07
#ifndef VMVN_H
#define VMVN_H

#include <neon_emu_types.h>

int8x8_t vmvn_s8(int8x8_t a) {
    int8x8_t r;
    int8x8_t tmp;
    tmp.v.i8 = __msa_fill_b(0xff);
    r.v.u8 = __msa_xor_v(a.v.u8, tmp.v.u8);
    return r;
}

#endif  // VMVN_H
