// 2023-04-21 14:01
#include <neon.h>
#include <neon_test.h>
// int8x8_t vget_high_s8(int8x16_t a)
// int16x4_t vget_high_s16(int16x8_t a)
// int32x2_t vget_high_s32(int32x4_t a)
// int64x1_t vget_high_s64(int64x2_t a)
// uint8x8_t vget_high_u8(uint8x16_t a)
// uint16x4_t vget_high_u16(uint16x8_t a)
// uint32x2_t vget_high_u32(uint32x4_t a)
// uint64x1_t vget_high_u64(uint64x2_t a)
// poly64x1_t vget_high_p64(poly64x2_t a)
// float16x4_t vget_high_f16(float16x8_t a)
// float32x2_t vget_high_f32(float32x4_t a)
// poly8x8_t vget_high_p8(poly8x16_t a)
// poly16x4_t vget_high_p16(poly16x8_t a)
// float64x1_t vget_high_f64(float64x2_t a)
//
// int8x8_t vget_low_s8(int8x16_t a)
// int16x4_t vget_low_s16(int16x8_t a)
// int32x2_t vget_low_s32(int32x4_t a)
// int64x1_t vget_low_s64(int64x2_t a)
// uint8x8_t vget_low_u8(uint8x16_t a)
// uint16x4_t vget_low_u16(uint16x8_t a)
// uint32x2_t vget_low_u32(uint32x4_t a)
// uint64x1_t vget_low_u64(uint64x2_t a)
// poly64x1_t vget_low_p64(poly64x2_t a)
// float16x4_t vget_low_f16(float16x8_t a)
// float32x2_t vget_low_f32(float32x4_t a)
// poly8x8_t vget_low_p8(poly8x16_t a)
// poly16x4_t vget_low_p16(poly16x8_t a)
// float64x1_t vget_low_f64(float64x2_t a)

TEST_CASE(test_vget_high_s8) {
    struct {
        int8_t a[16];
        int8_t r[8];
    } test_vec[] = {
        {{2, -102, -10, -48, -126, -61, 29, 91, -59, -8, 97, 27, -101, 40, -4,
          34},
         {-59, -8, 97, 27, -101, 40, -4, 34}},
        {{-127, -21, -97, -6, 83, 58, -124, -3, 103, 78, -13, -42, -66, 46, 62,
          -64},
         {103, 78, -13, -42, -66, 46, 62, -64}},
        {{-56, 52, -112, 74, -9, -83, -90, -68, -91, 7, -41, 64, 47, -44, 98,
          -79},
         {-91, 7, -41, 64, 47, -44, 98, -79}},
        {{-65, 1, -85, 18, 60, 47, 15, -93, 126, 3, 121, 60, 49, -73, -3, -6},
         {126, 3, 121, 60, 49, -73, -3, -6}},
        {{-21, -115, 68, -29, 59, -22, -97, -32, -14, 119, 33, 33, 75, -125,
          -46, 10},
         {-14, 119, 33, 33, 75, -125, -46, 10}},
        {{-123, 126, 29, -63, -83, 44, 100, 43, 47, -35, 104, 97, -108, 101, 91,
          INT8_MAX},
         {47, -35, 104, 97, -108, 101, 91, INT8_MAX}},
        {{-14, -97, 98, 45, -118, 2, 14, 124, 121, 47, -99, -60, -78, 112, -50,
          55},
         {121, 47, -99, -60, -78, 112, -50, 55}},
        {{-18, -21, -8, -101, 24, 92, -57, 71, 57, 47, -88, -51, -108, 3, 77,
          -122},
         {57, 47, -88, -51, -108, 3, 77, -122}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x16_t a = vld1q_s8(test_vec[i].a);
        int8x8_t r = vget_high_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
