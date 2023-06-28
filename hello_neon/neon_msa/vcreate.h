// 2023-04-21 11:27
#ifndef VCREATE_H
#define VCREATE_H

#include <neon_emu_types.h>

// NOTE: msa 需要 128 bit, 所以这段代码可能会出错, 因为紧接着 a 的地址可能无法访
// 问
int8x8_t vcreate_s8(uint64_t a) {
    int8x8_t r;
    r.v.i8 = __msa_ld_b(&a, 0);
    return r;
}

#endif  // VCREATE_H
