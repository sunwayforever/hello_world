// 2023-04-20 17:37
#ifndef VORN_H
#define VORN_H

#include <neon_emu_types.h>
#include <vmvn.h>

int8x8_t vorn_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int8x8_t tmp = vmvn_s8(b);
    r.v.u8 = __msa_or_v(a.v.u8, tmp.v.u8);
    return r;
}

#endif  // VORN_H
