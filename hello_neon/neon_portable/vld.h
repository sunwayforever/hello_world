// 2023-04-14 13:07
#ifndef VLD_H
#define VLD_H
#include <neon_emu_types.h>

#define DEF_VLD1(base, sign, size, n)                                   \
    base##size##x##n##_t vld1_##sign##size(const base##size##_t* ptr) { \
        base##size##x##n##_t r;                                         \
        memcpy(&r, ptr, sizeof(r.values));                              \
        return r;                                                       \
    }

#define DEF_VLD1Q(base, sign, size, n)                                   \
    base##size##x##n##_t vld1q_##sign##size(const base##size##_t* ptr) { \
        base##size##x##n##_t r;                                          \
        memcpy(&r, ptr, sizeof(r.values));                               \
        return r;                                                        \
    }

#define DEF_VLD1_LANE(base, sign, size, n)                             \
    base##size##x##n##_t vld1_lane_##sign##size(                       \
        const base##size##_t* ptr, base##size##x##n##_t v, int lane) { \
        base##size##x##n##_t r = v;                                    \
        r.values[lane] = *ptr;                                         \
        return r;                                                      \
    }

DEF_VLD1(int, s, 8, 8);
DEF_VLD1(uint, u, 8, 8);
DEF_VLD1(int, s, 16, 4);
DEF_VLD1(uint, u, 16, 4);
DEF_VLD1(int, s, 32, 2);
DEF_VLD1(uint, u, 32, 2);
DEF_VLD1(int, s, 64, 1);
DEF_VLD1(uint, u, 64, 1);
DEF_VLD1(float, f, 32, 2);
DEF_VLD1(float, f, 64, 1);

DEF_VLD1Q(int, s, 8, 16);
DEF_VLD1Q(uint, u, 8, 16);
DEF_VLD1Q(int, s, 16, 8);
DEF_VLD1Q(uint, u, 16, 8);
DEF_VLD1Q(int, s, 32, 4);
DEF_VLD1Q(uint, u, 32, 4);
DEF_VLD1Q(int, s, 64, 2);
DEF_VLD1Q(uint, u, 64, 2);
DEF_VLD1Q(float, f, 32, 4);
DEF_VLD1Q(float, f, 64, 2);

DEF_VLD1_LANE(int, s, 8, 8);
#endif  // VLD_H
