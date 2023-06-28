// 2023-04-20 17:33
#ifndef VEOR_H
#define VEOR_H

#include <neon_emu_types.h>

int8x8_t veor_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.u8 = __msa_xor_v(a.v.u8, b.v.u8);
    return r;
}

#endif  // VEOR_H
