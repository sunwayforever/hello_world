// 2023-04-20 17:07
#ifndef VMVN_H
#define VMVN_H

#include <neon_emu_types.h>

int8x8_t vmvn_s8(int8x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = ~a.values[i];
    }
    return r;
}

#endif  // VMVN_H
