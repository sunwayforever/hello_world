// 2023-04-18 11:57
#ifndef VPADD_H
#define VPADD_H

#include <neon_emu_types.h>

// NOTE: 低效的 pair 操作
int8x8_t vpadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int8x8_t _a, _b;
    int8x16_t _c;
    MERGE(_c, a, b);
    _a.v.i8 = __msa_pckev_b(_c.v.i8, _c.v.i8);
    _b.v.i8 = __msa_pckod_b(_c.v.i8, _c.v.i8);
    r.v.i8 = __msa_addv_b(_a.v.i8, _b.v.i8);
    return r;
}

int16x4_t vpadal_s8(int16x4_t a, int8x8_t b) {
    int16x4_t r;
    int16x8_t _b;
    COPY(_b, b);

    int16x8_t _x, _y;
    _x.v.i16 = __msa_pckev_h(_b.v.i16, _b.v.i16);
    _y.v.i16 = __msa_pckod_h(_b.v.i16, _b.v.i16);
    r.v.i16 = __msa_addv_h(_x.v.i16, _y.v.i16);

    r.v.i16 = __msa_addv_h(r.v.i16, a.v.i16);
    return r;
}
#endif  // VPADD_H
