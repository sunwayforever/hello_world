// 2023-04-21 10:16
#ifndef VRBIT_H
#define VRBIT_H

#include <neon_emu_types.h>

int8x8_t vrbit_s8(int8x8_t a) {
    // NOTE:
    // abcdefgh -> badcfehg
    // num = (((num & 0xaa) >> 1) | ((num & 0x55) << 1));
    // badcfehg -> dcbahgfe
    // num = (((num & 0xcc) >> 2) | ((num & 0x33) << 2));
    // dcbahgfe -> hgfedcba
    // num = (((num & 0xf0) >> 4) | ((num & 0x0f) << 4));

    int8x8_t r = a;

    int8x8_t _x, _y;

    // addi(u) -> u
    // srli (i) -> i
    // orv(u) -> u
    _x.v.u8 = __msa_andi_b(r.v.u8, 0xaa);
    _y.v.u8 = __msa_andi_b(r.v.u8, 0x55);
    _x.v.i8 = __msa_srli_b(_x.v.i8, 1);
    _y.v.i8 = __msa_slli_b(_y.v.i8, 1);
    r.v.u8 = __msa_or_v(_x.v.u8, _y.v.u8);

    _x.v.u8 = __msa_andi_b(r.v.u8, 0xcc);
    _y.v.u8 = __msa_andi_b(r.v.u8, 0x33);
    _x.v.i8 = __msa_srli_b(_x.v.i8, 2);
    _y.v.i8 = __msa_slli_b(_y.v.i8, 2);
    r.v.u8 = __msa_or_v(_x.v.u8, _y.v.u8);

    _x.v.u8 = __msa_andi_b(r.v.u8, 0xf0);
    _y.v.u8 = __msa_andi_b(r.v.u8, 0x0f);
    _x.v.i8 = __msa_srli_b(_x.v.i8, 4);
    _y.v.i8 = __msa_slli_b(_y.v.i8, 4);
    r.v.u8 = __msa_or_v(_x.v.u8, _y.v.u8);

    return r;
}

#endif  // VRBIT_H
