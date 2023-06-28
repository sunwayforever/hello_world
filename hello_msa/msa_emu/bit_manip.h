// 2023-04-27 17:39
#ifndef BIT_MANIP_H
#define BIT_MANIP_H

#include "msa_emu_types.h"

v16u8 __msa_bclr_b(v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        int mod = b.values[i] % 8;
        if (mod < 0) {
            mod += 8;
        }
        r.values[i] = a.values[i] & ~(1 << mod);
    }
    return r;
}

v16u8 __msa_bset_b(v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        int mod = b.values[i] % 8;
        if (mod < 0) {
            mod += 8;
        }
        r.values[i] = a.values[i] | (1 << mod);
    }
    return r;
}

v16u8 __msa_binsl_b(v16u8 c, v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        int n = b.values[i] % 8;
        if (n < 0) {
            n += 8;
        }
        uint8_t mask = (1 << (8 - n)) - 1;
        r.values[i] = (c.values[i] & mask) | (a.values[i] & ~mask);
    }
    return r;
}

v16u8 __msa_bmnz_v(v16u8 c, v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] =
            (a.values[i] & b.values[i]) | (c.values[i] & ~b.values[i]);
    }
    return r;
}

v16u8 __msa_bsel_v(v16u8 c, v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] =
            (a.values[i] & ~c.values[i]) | (b.values[i] & c.values[i]);
    }
    return r;
}

v16i8 __msa_nloc_b(v16i8 a) {
    v16i8 r = {0};
    for (int i = 0; i < 16; i++) {
        uint8_t tmp = a.values[i];
        for (int j = 0; j < 8; j++) {
            if ((tmp & (1 << 7)) == 0) {
                break;
            }
            r.values[i] += 1;
            tmp <<= 1;
        }
    }
    return r;
}

v16i8 __msa_nlzc_b(v16i8 a) {
    v16i8 r = {0};
    for (int i = 0; i < 16; i++) {
        uint8_t tmp = a.values[i];
        for (int j = 0; j < 8; j++) {
            if ((tmp & (1 << 7)) != 0) {
                break;
            }
            r.values[i] += 1;
            tmp <<= 1;
        }
    }
    return r;
}

#endif  // BIT_MANIP_H
