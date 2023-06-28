// 2023-04-20 18:57
#ifndef VBSL_H
#define VBSL_H

#include <neon_emu_types.h>
int8x8_t vbsl_s8(uint8x8_t mask, int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.u8 = __msa_bsel_v(mask.v.u8, b.v.u8, a.v.u8);
    return r;
}

#endif  // VBSL_H
