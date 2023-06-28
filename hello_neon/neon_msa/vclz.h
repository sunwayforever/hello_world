// 2023-04-20 17:58
#ifndef VCLZ_H
#define VCLZ_H

#include <neon_emu_types.h>

int8x8_t vclz_s8(int8x8_t a) {
    int8x8_t r;
    r.v.i8 = __msa_nlzc_b(a.v.i8);
    return r;
}

#endif  // VCLZ_H
