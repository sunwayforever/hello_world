// 2023-04-18 17:01
#ifndef VCGT_H
#define VCGT_H

#include <neon_emu_types.h>

uint8x8_t vcgt_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    r.v.i8 = __msa_clt_s_b(b.v.i8, a.v.i8);
    return r;
}
#endif  // VCGT_H
