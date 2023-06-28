// 2023-04-27 15:47
#ifndef DOTP_H
#define DOTP_H

#include "msa_emu_types.h"

v8i16 __msa_dotp_s_h(v16i8 a, v16i8 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = a.values[2 * i + 1] * b.values[2 * i + 1] +
                      a.values[2 * i] * b.values[2 * i];
    }
    return r;
}

v8i16 __msa_dpadd_s_h(v8i16 c, v16i8 a, v16i8 b) {
    v8i16 r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = c.values[i] + a.values[2 * i + 1] * b.values[2 * i + 1] +
                      a.values[2 * i] * b.values[2 * i];
    }
    return r;
}

#endif  // DOTP_H
