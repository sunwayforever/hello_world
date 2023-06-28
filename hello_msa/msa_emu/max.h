// 2023-04-27 16:11
#ifndef MAX_H
#define MAX_H

#include <assert.h>

#include "msa_emu_types.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))

v16i8 __msa_max_s_b(v16i8 a, v16i8 b) {
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = MAX(a.values[i], b.values[i]);
    }
    return r;
}

v16i8 __msa_maxi_s_b(v16i8 a, int b) {
    assert(b >= -16 && b <= 15);
    v16i8 r;
    for (int i = 0; i < 16; i++) {
        r.values[i] = MAX(a.values[i], b);
    }
    return r;
}
#endif  // MAX_H
