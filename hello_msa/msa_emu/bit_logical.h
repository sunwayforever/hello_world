// 2023-04-27 16:52
#ifndef BIT_H
#define BIT_H

#include "msa_emu_types.h"

v16u8 __msa_and_v(v16u8 a, v16u8 b) {
    v16u8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[i] & b.values[i];
    }
    return r;
}
#endif  // BIT_H
