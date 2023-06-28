// 2023-04-18 16:19
#ifndef VCEQ_H
#define VCEQ_H

#include <neon_emu_types.h>

uint8x8_t vceq_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    r.v.i8 = __msa_ceq_b(a.v.i8, b.v.i8);
    return r;
}
#endif  // VCEQ_H
