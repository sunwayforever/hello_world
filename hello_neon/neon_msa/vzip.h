// 2023-04-21 15:58
#ifndef VZIP_H
#define VZIP_H

#include <neon_emu_types.h>

int8x8_t vzip1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_ilvr_b(b.v.i8, a.v.i8);
    return r;
}

// NOTE: vzip2 要求从 offset+4 位置开始 copy, 而 ilvl 只能从 offset+8 位置开始.
// 如果该测试操作的数据是 int8x16_t, 则可以使用 ilvl, 否则只能使用复杂的 shuffle
int8x8_t vzip2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    int8x8_t shuffle;
    int8_t tmp[] = {4, 20, 5, 21, 6, 22, 7, 23};
    shuffle.v.i8 = __msa_ld_b(tmp, 0);
    r.v.i8 = __msa_vshf_b(shuffle.v.i8, b.v.i8, a.v.i8);
    return r;
}

int8x8x2_t vzip_s8(int8x8_t a, int8x8_t b) {
    int8x8x2_t r;

    int8x8_t shuffle;
    int8_t tmp[] = {0, 16, 1, 17, 2, 18, 3, 19};
    shuffle.v.i8 = __msa_ld_b(tmp, 0);
    r.val[0].v.i8 = __msa_vshf_b(shuffle.v.i8, b.v.i8, a.v.i8);

    int8_t tmp2[] = {4, 20, 5, 21, 6, 22, 7, 23};
    shuffle.v.i8 = __msa_ld_b(tmp2, 0);
    r.val[1].v.i8 = __msa_vshf_b(shuffle.v.i8, b.v.i8, a.v.i8);

    return r;
}

#endif  // VZIP_H
