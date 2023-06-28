// 2023-04-17 15:08
#ifndef REINTERP_H
#define REINTERP_H

#include <neon_emu_types.h>

#define DEF_REINTERP(to, to_short, from, from_short)    \
    to vreinterpret_##to_short##_##from_short(from a) { \
        to r;                                           \
        memcpy(&r, &a, sizeof(r));                      \
        return r;                                       \
    }

DEF_REINTERP(poly8x8_t, p8, uint8x8_t, u8);
DEF_REINTERP(float32x2_t, f32, int8x8_t, s8);
DEF_REINTERP(int8x8_t, s8, float32x2_t, f32);
DEF_REINTERP(uint64x1_t, u64, int8x8_t, s8);
DEF_REINTERP(int8x8_t, s8, uint8x8_t, u8);
#endif  // REINTERP_H
