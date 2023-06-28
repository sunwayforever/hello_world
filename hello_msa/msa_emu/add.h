// 2023-04-26 14:41
#ifndef ADD_H
#define ADD_H

#include <assert.h>
#include <stdint.h>

#include "msa_emu_types.h"

v16i8 __msa_addv_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] + b.values[i];
    }
    return r;
}

v16i8 __msa_addvi_b(v16i8 a, uint8_t b) {
    assert(b < 32);
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] + b;
    }
    return r;
}

#define ABS(x) ((x) > 0 ? (x) : -(x))
v16i8 __msa_add_a_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = ABS(a.values[i]) + ABS(b.values[i]);
    }
    return r;
}

v16i8 __msa_adds_a_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        int16_t tmp = ABS(a.values[i]) + ABS(b.values[i]);
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        } else if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = tmp;
    }
    return r;
}

v16i8 __msa_adds_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        int16_t tmp = a.values[i] + b.values[i];
        if (tmp > INT8_MAX) {
            tmp = INT8_MAX;
        } else if (tmp < INT8_MIN) {
            tmp = INT8_MIN;
        }
        r.values[i] = tmp;
    }
    return r;
}

v16u8 __msa_adds_u_b(v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        int16_t tmp = a.values[i] + b.values[i];
        if (tmp > UINT8_MAX) {
            tmp = UINT8_MAX;
        }
        r.values[i] = tmp;
    }
    return r;
}

v8i16 __msa_hadd_s_h(v16i8 a, v16i8 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[2 * i + 1] + b.values[2 * i];
    }
    return r;
}

v16i8 __msa_asub_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = ABS(a.values[i] - b.values[i]);
    }
    return r;
}

v16i8 __msa_maddv_b(v16i8 c, v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = c.values[i] + a.values[i] * b.values[i];
    }
    return r;
}

v16i8 __msa_sat_s_b(v16i8 a, int n) {
    assert(n >= 0 && n <= 7);
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        int8_t tmp = a.values[i];
        if (tmp > ((1 << n) - 1)) {
            tmp = (1 << n) - 1;
        } else if (tmp < -((1 << n))) {
            tmp = -(1 << n);
        }
        r.values[i] = tmp;
    }
    return r;
}

#endif  // ADD_H
