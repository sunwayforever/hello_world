// 2023-04-20 17:19
#ifndef VORR_H
#define VORR_H

#include <neon_emu_types.h>

int8x8_t vorr_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.u8 = __msa_or_v(a.v.u8, b.v.u8);
    return r;
}
#endif  // VORR_H
