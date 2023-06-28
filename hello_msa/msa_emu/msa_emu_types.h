// 2023-04-14 13:08
#ifndef ARM_NEON_COMMON_H
#define ARM_NEON_COMMON_H

#include <stdint.h>
#include <string.h>

typedef float float32_t;
typedef double float64_t;

#define DEF_TYPE(base, n, abbr) \
    typedef union {             \
        base##_t values[n];     \
    } v##n##abbr;

DEF_TYPE(int8, 16, i8);
DEF_TYPE(uint8, 16, u8);
DEF_TYPE(int16, 8, i16);
DEF_TYPE(uint16, 8, u16);
DEF_TYPE(int32, 4, i32);
DEF_TYPE(uint32, 4, u32);
DEF_TYPE(float32, 4, f32);
DEF_TYPE(float64, 2, f64);

#endif  // ARM_NEON_COMMON_H
