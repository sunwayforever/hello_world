// 2023-04-20 17:45
#ifndef VCLS_H
#define VCLS_H

#include <neon_emu_types.h>
int8x8_t vcls_s8(int8x8_t a) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        uint8_t tmp = a.values[i];
        uint8_t sign = tmp & (1 << 7);
        for (int j = 0; j < 8; j++) {
            if ((tmp & (1 << 7)) != sign) {
                break;
            }
            tmp <<= 1;
            r.values[i] += 1;
        }
        r.values[i]--;
    }
    return r;
}

int8x8_t vcls_u8(uint8x8_t a) {
    int8x8_t r = {0};
    for (int i = 0; i < 8; i++) {
        uint8_t tmp = a.values[i];
        uint8_t sign = tmp & (1 << 7);
        for (int j = 0; j < 8; j++) {
            if ((tmp & (1 << 7)) != sign) {
                break;
            }
            tmp <<= 1;
            r.values[i] += 1;
        }
        r.values[i]--;
    }
    return r;
}

#endif  // VCLS_H
