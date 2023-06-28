// 2023-04-21 13:43
#ifndef VDUP_H
#define VDUP_H

#include <neon_emu_types.h>

int8x8_t vdup_n_s8(int8_t a) {
    int8x8_t r;
    r.v.i8 = __msa_fill_b(a);
    return r;
}
#endif  // VDUP_H
