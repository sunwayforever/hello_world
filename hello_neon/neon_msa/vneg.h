// 2023-04-20 16:45
#ifndef VNEG_H
#define VNEG_H
#include <neon_emu_types.h>

int8x16_t vnegq_s8(int8x16_t a) {
    int8x16_t r;
    int8x16_t zero;
    zero.v.i8 = __msa_fill_b(0);
    r.v.i8 = __msa_subv_b(zero.v.i8, a.v.i8);
    return r;
}

int8x8_t vqneg_s8(int8x8_t a) {
    int8x8_t r;
    int8x16_t zero;
    zero.v.i8 = __msa_fill_b(0);
    r.v.i8 = __msa_subs_s_b(zero.v.i8, a.v.i8);
    return r;
}

int8_t vqnegb_s8(int8_t a) {
    if (a == INT8_MIN) {
        return INT8_MAX;
    }
    return -a;
}
#endif  // VNEG_H
