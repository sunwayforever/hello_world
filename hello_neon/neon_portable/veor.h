// 2023-04-20 17:33
#ifndef VEOR_H
#define VEOR_H

#include <neon_emu_types.h>

int8x8_t veor_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] ^ b.values[i];
    }
    return r;
}

#endif  // VEOR_H
