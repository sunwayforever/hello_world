// 2023-04-27 16:00
#ifndef DIV_H
#define DIV_H

#include "msa_emu_types.h"

v16i8 __msa_div_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] / b.values[i];
    }
    return r;
}

#endif  // DIV_H
