// 2023-04-27 16:59
#ifndef BIT_SHIFT_H
#define BIT_SHIFT_H

#include <stdint.h>

#include "msa_emu_types.h"

v8i16 __msa_sll_h(v8i16 a, v8i16 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        int mod = b.values[i] % 16;
        if (mod < 0) {
            mod += 16;
        }
        r.values[i] = a.values[i] << mod;
    }
    return r;
}

v8i16 __msa_sra_h(v8i16 a, v8i16 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        int mod = b.values[i] % 16;
        if (mod < 0) {
            mod += 16;
        }
        r.values[i] = a.values[i] >> mod;
    }
    return r;
}

v8i16 __msa_srl_h(v8i16 a, v8i16 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        int mod = b.values[i] % 16;
        if (mod < 0) {
            mod += 16;
        }
        r.values[i] = (uint16_t)a.values[i] >> mod;
    }
    return r;
}

v8i16 __msa_srlr_h(v8i16 a, v8i16 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        int mod = b.values[i] % 16;
        if (mod < 0) {
            mod += 16;
        }
        r.values[i] = ((uint16_t)a.values[i] + (1 << (mod - 1))) >> mod;
    }
    return r;
}

#endif  // BIT_SHIFT_H
