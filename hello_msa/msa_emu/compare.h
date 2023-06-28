// 2023-04-28 16:22
#ifndef COMPARE_H
#define COMPARE_H

#include "msa_emu_types.h"

v16i8 __msa_clt_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        if (a.values[i] < b.values[i]) {
            r.values[i] = -1;
        } else {
            r.values[i] = 0;
        }
    }
    return r;
}

#endif  // COMPARE_H
