// 2023-04-21 10:16
#ifndef VRBIT_H
#define VRBIT_H

#include <stdint.h>

#include <neon_emu_types.h>

int8x8_t vrbit_s8(int8x8_t a) {
    int8x8_t r;
    for (int i = 0; i < 8; i++) {
        int8_t _r = 0;
        int8_t _a = a.values[i];
        for (int j = 0; j < 8; j++) {
            _r <<= 1;
            _r |= _a & 1;
            _a >>= 1;
        }
        r.values[i] = _r;
    }
    return r;
}

#endif  // VRBIT_H
