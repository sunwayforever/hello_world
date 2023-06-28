// 2023-04-20 18:15
#ifndef VCNT_H
#define VCNT_H

#include <neon_emu_types.h>

int8x8_t vcnt_s8(int8x8_t a) {
    int8x8_t r;
    r.v.i8 = __msa_pcnt_b(a.v.i8);
    return r;
}

#endif  // VCNT_H
