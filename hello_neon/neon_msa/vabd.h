// 2023-04-17 15:40
#ifndef VABD_H
#define VABD_H

#include <neon_emu_types.h>

int8x8_t vabd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_asub_s_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vaba_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r, tmp;
    tmp.v.i8 = __msa_asub_s_b(b.v.i8, c.v.i8);
    r.v.i8 = __msa_addv_b(a.v.i8, tmp.v.i8);
    return r;
}

int8x8_t vabs_s8(int8x8_t a) {
    int8x8_t r;
    int8x8_t zero;
    zero.v.i8 = __msa_fill_b(0);
    r.v.i8 = __msa_add_a_b(a.v.i8, zero.v.i8);
    return r;
}

// NOTE: msa 不支持标量
int8_t vqabsb_s8(int8_t a) {
    int16_t tmp;
    tmp = a > 0 ? a : -a;
    if (tmp > INT8_MAX) {
        tmp = INT8_MAX;
    }
    return (int8_t)tmp;
}

#endif  // VABD_H
