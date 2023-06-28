// 2023-04-20 18:42
#include <neon.h>
#include <neon_test.h>
// int8x8_t vbic_s8(int8x8_t a,int8x8_t b)
// int16x4_t vbic_s16(int16x4_t a,int16x4_t b)
// int32x2_t vbic_s32(int32x2_t a,int32x2_t b)
// int64x1_t vbic_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vbic_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vbic_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vbic_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vbic_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t vbicq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vbicq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vbicq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vbicq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vbicq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vbicq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vbicq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vbicq_u64(uint64x2_t a,uint64x2_t b)

TEST_CASE(test_vbic_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-62, 9, 113, 100, -96, -44, 44, -52},
         {-99, -76, 33, -61, -16, -42, 93, 44},
         {66, 9, 80, 36, 0, 0, 32, -64}},
        {{1, -113, 51, -4, -32, 91, 93, -121},
         {52, 72, 55, -13, -124, 83, -76, 70},
         {1, -121, 0, 12, 96, 8, 73, -127}},
        {{92, 37, -85, -3, -6, -41, -55, -105},
         {-117, -22, 91, 123, -64, -72, -89, -63},
         {84, 5, -96, -124, 58, 71, 72, 22}},
        {{71, -38, -66, 39, 54, 27, -81, 106},
         {99, -26, 93, -25, 58, 18, 45, -106},
         {4, 24, -94, 0, 4, 9, -126, 104}},
        {{55, -40, -109, 49, -81, 92, -55, 58},
         {71, 36, -75, 7, -36, 93, -55, 36},
         {48, -40, 2, 48, 35, 0, 0, 26}},
        {{55, -121, 75, 109, -94, -6, -40, 5},
         {-31, 53, -20, 27, 71, 25, -79, INT8_MAX},
         {22, -126, 3, 100, -96, -30, 72, 0}},
        {{-14, 69, -80, -95, -95, 121, -36, -24},
         {-99, -111, -16, 122, -18, -71, -98, 38},
         {98, 68, 0, -127, 1, 64, 64, -56}},
        {{64, -23, -109, -30, -28, 107, -25, -59},
         {-95, -45, -32, -24, -20, -111, 103, -34},
         {64, 40, 19, 2, 0, 106, INT8_MIN, 1}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vbic_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
