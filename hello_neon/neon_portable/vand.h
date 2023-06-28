// 2023-04-20 17:09
#ifndef VAND_H
#define VAND_H

#include <neon_emu_types.h>

int8x8_t vand_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] & b.values[i];
    }
    return r;
}
#endif  // VAND_H
