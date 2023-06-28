// 2023-04-20 17:19
#ifndef VORR_H
#define VORR_H

#include <neon_emu_types.h>

int8x8_t vorr_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i] | b.values[i];
    }
    return r;
}
#endif  // VORR_H
