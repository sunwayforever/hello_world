// 2023-05-04 10:59
#ifndef VAND_H
#define VAND_H

#include <neon_emu_types.h>

int8x8_t vand_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.u8 = __msa_and_v(a.v.u8, b.v.u8);
    return r;
}

#endif  // VAND_H
