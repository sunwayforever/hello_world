// 2023-04-27 15:16
#ifndef AVE_H
#define AVE_H

#include "msa_emu_types.h"

v16i8 __msa_ave_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = ((int16_t)a.values[i] + (int16_t)b.values[i]) >> 1;
    }
    return r;
}

v16i8 __msa_aver_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = ((int16_t)a.values[i] + (int16_t)b.values[i] + 1) >> 1;
    }
    return r;
}

#endif  // AVE_H
