// 2023-04-21 15:22
#ifndef VEXT_H
#define VEXT_H

#include <neon_emu_types.h>

int8x8_t vext_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    for (int i = n; i < 8 + n; i++) {
        if (i < 8) {
            r.values[i - n] = a.values[i];
        } else {
            r.values[i - n] = b.values[i - 8];
        }
    }
    return r;
}
#endif  // VEXT_H
