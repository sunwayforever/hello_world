// 2023-04-21 13:58
#ifndef VCOMBINE_H
#define VCOMBINE_H

#include <neon_emu_types.h>

int8x16_t vcombine_s8(int8x8_t a, int8x8_t b) {
    int8x16_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[i];
        r.values[i + 8] = b.values[i];
    }
    return r;
}
#endif  // VCOMBINE_H
