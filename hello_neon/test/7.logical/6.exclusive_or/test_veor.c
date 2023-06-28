// 2023-04-20 17:20
#include <neon.h>
#include <neon_test.h>
// int8x8_t veor_s8(int8x8_t a,int8x8_t b)
// int16x4_t veor_s16(int16x4_t a,int16x4_t b)
// int32x2_t veor_s32(int32x2_t a,int32x2_t b)
// int64x1_t veor_s64(int64x1_t a,int64x1_t b)
// uint8x8_t veor_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t veor_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t veor_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t veor_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t veorq_s8(int8x16_t a,int8x16_t b)
// int16x8_t veorq_s16(int16x8_t a,int16x8_t b)
// int32x4_t veorq_s32(int32x4_t a,int32x4_t b)
// int64x2_t veorq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t veorq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t veorq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t veorq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t veorq_u64(uint64x2_t a,uint64x2_t b)

TEST_CASE(test_veor_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-57, 76, 112, 126, 46, -66, -3, 47},
         {8, -105, -32, -33, 118, 37, 95, 95},
         {-49, -37, -112, -95, 88, -101, -94, 112}},
        {{-52, -65, 78, -24, 60, -59, -102, -44},
         {-22, -3, 118, -6, 21, -127, 40, -36},
         {38, 66, 56, 18, 41, 68, -78, 8}},
        {{-50, -104, 90, -4, 86, 88, 43, 94},
         {-17, 11, 62, 101, 48, -99, -60, -3},
         {33, -109, 100, -103, 102, -59, -17, -93}},
        {{92, 19, -27, -103, -40, INT8_MAX, 109, -61},
         {124, -28, -67, -110, 101, -26, 110, 51},
         {32, -9, 88, 11, -67, -103, 3, -16}},
        {{126, -55, 48, -43, 33, 91, 51, 16},
         {103, 113, 117, -105, 14, 58, -108, 107},
         {25, -72, 69, 66, 47, 97, -89, 123}},
        {{77, 122, 4, 37, -7, 113, -24, 118},
         {85, -90, 8, -69, -116, 118, -18, 10},
         {24, -36, 12, -98, 117, 7, 6, 124}},
        {{63, 30, -33, 96, 122, 19, 112, -31},
         {-124, -26, 120, -109, 32, 13, -2, 109},
         {-69, -8, -89, -13, 90, 30, -114, -116}},
        {{-121, 2, -110, INT8_MIN, 115, 123, -10, -55},
         {33, -2, -124, -83, 117, 114, -73, -76},
         {-90, -4, 22, 45, 6, 9, 65, 125}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = veor_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
