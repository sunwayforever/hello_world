// 2023-04-21 15:22
#ifndef VEXT_H
#define VEXT_H

#include <neon_emu_types.h>

// NOTE: 由于 a,b 后均有 8 个 0, 导致直接使用 sld 会出错
int8x8_t vext_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    int8x16_t _c;
    MERGE(_c, a, b);
    r.v.i8 = __msa_sld_b(_c.v.i8, _c.v.i8, n);
    return r;
}
#endif  // VEXT_H
