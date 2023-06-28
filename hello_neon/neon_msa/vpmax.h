// 2023-04-18 14:24
#ifndef VPMAX_H
#define VPMAX_H

#include <neon_emu_types.h>

// NOTE: pair 操作因为 vector 末尾的 0 而很低效
int8x8_t vpmax_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int8x16_t _c;
    MERGE(_c, a, b);
    int8x8_t _a, _b;
    _a.v.i8 = __msa_pckev_b(_c.v.i8, _c.v.i8);
    _b.v.i8 = __msa_pckod_b(_c.v.i8, _c.v.i8);
    r.v.i8 = __msa_max_s_b(_a.v.i8, _b.v.i8);
    return r;
}

#endif  // VPMAX_H
