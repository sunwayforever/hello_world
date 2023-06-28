// 2023-04-28 17:03
#ifndef SHUFFLE_H
#define SHUFFLE_H

#include "msa_emu_types.h"

v16i8 __msa_ilvev_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 8; i++) {
        r.values[2 * i] = b.values[2 * i];
        r.values[2 * i + 1] = a.values[2 * i];
    }
    return r;
}

v16i8 __msa_ilvl_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 8; i++) {
        r.values[2 * i] = b.values[i + 8];
        r.values[2 * i + 1] = a.values[i + 8];
    }
    return r;
}

v16i8 __msa_pckev_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 8; i++) {
        r.values[i] = b.values[2 * i];
        r.values[i + 8] = a.values[2 * i];
    }
    return r;
}

v16i8 __msa_vshf_b(v16i8 c, v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        int index = c.values[i];
        if (index > 15) {
            r.values[i] = a.values[index - 16];
        } else {
            r.values[i] = b.values[index];
        }
    }
    return r;
}

v16i8 __msa_sld_b(v16i8 a, v16i8 b, int n) {
    v16i8 r;
    int offset = n % 16;
    if (offset < 0) offset += 16;
    for (int i = 0; i < 16; i++) {
        int index = offset + i;
        if (index > 15) {
            r.values[i] = a.values[index - 16];
        } else {
            r.values[i] = b.values[index];
        }
    }
    return r;
}

#endif  // SHUFFLE_H
