// 2023-04-26 16:46
#ifndef LD_H
#define LD_H

#include "msa_emu_types.h"

v16i8 __msa_ld_b(void *a, int n) {
    v16i8 r;
    memcpy(&r, a, 16);
    return r;
}

v8i16 __msa_ld_h(void *a, int n) {
    v8i16 r;
    memcpy(&r, a, 16);
    return r;
}

v4i32 __msa_ld_w(void *a, int n) {
    v4i32 r;
    memcpy(&r, a, 16);
    return r;
}

v16i8 __msa_splat_b(v16i8 a, int b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = a.values[b];
    }
    return r;
}
#endif  // LD_H
