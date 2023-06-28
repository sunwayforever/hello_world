// 2023-04-18 16:19
#ifndef VCEQ_H
#define VCEQ_H

#include <neon_emu_types.h>

uint8x8_t vceq_s8(int8x8_t a, int8x8_t b) {
    uint8x8_t r;
    for (int i = 0; i < 8; i++) {
        if (memcmp(&a.values[i], &b.values[i], sizeof(a.values[0])) == 0) {
            r.values[i] = UINT8_MAX;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}
#endif  // VCEQ_H
