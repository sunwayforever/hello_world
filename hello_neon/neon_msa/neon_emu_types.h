// 2023-04-14 13:08
#ifndef NEON_EMU_TYPES_H
#define NEON_EMU_TYPES_H

#include <msa.h>
#include <stdint.h>
#include <string.h>

typedef float float32_t;
typedef double float64_t;

typedef union {
    v16i8 i8;
    v16u8 u8;
    v8i16 i16;
    v8u16 u16;
    v4i32 i32;
    v4u32 u32;
    v4f32 f32;
    v2f64 f64;
} m128;

#define DEF_TYPE(base, n)   \
    typedef union {         \
        m128 v;             \
        base##_t values[n]; \
    } base##x##n##_t;

#define DEF_ARRAY_TYPE(base, n) \
    typedef union {             \
        m128 v[n];              \
        base##_t val[n];        \
    } base##x##n##_t;

DEF_TYPE(int8, 8);
DEF_TYPE(int8, 16);
DEF_TYPE(uint8, 8);
DEF_TYPE(uint8, 16);

DEF_TYPE(int16, 4);
DEF_TYPE(int16, 8);
DEF_TYPE(uint16, 4);
DEF_TYPE(uint16, 8);

DEF_TYPE(int32, 2);
DEF_TYPE(int32, 4);
DEF_TYPE(uint32, 2);
DEF_TYPE(uint32, 4);

DEF_TYPE(int64, 1);
DEF_TYPE(int64, 2);
DEF_TYPE(uint64, 1);
DEF_TYPE(uint64, 2);

DEF_TYPE(float32, 2);
DEF_TYPE(float32, 4);
DEF_TYPE(float64, 1);
DEF_TYPE(float64, 2);

typedef union {
    uint8_t values[8];
} poly8x8_t;

DEF_ARRAY_TYPE(int8x8, 2);

#endif  // NEON_EMU_TYPES_H
