// 2023-04-19 14:59
#include <neon.h>
#include <neon_test.h>
// int8x8_t vrshl_s8(int8x8_t a,int8x8_t b)
// int16x4_t vrshl_s16(int16x4_t a,int16x4_t b)
// int32x2_t vrshl_s32(int32x2_t a,int32x2_t b)
// int64x1_t vrshl_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vrshl_u8(uint8x8_t a,int8x8_t b)
// uint16x4_t vrshl_u16(uint16x4_t a,int16x4_t b)
// uint32x2_t vrshl_u32(uint32x2_t a,int32x2_t b)
// uint64x1_t vrshl_u64(uint64x1_t a,int64x1_t b)
//
// int8x16_t vrshlq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vrshlq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vrshlq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vrshlq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vrshlq_u8(uint8x16_t a,int8x16_t b)
// uint16x8_t vrshlq_u16(uint16x8_t a,int16x8_t b)
// uint32x4_t vrshlq_u32(uint32x4_t a,int32x4_t b)
// uint64x2_t vrshlq_u64(uint64x2_t a,int64x2_t b)
// -----------------------------------------------
// int64_t vrshld_s64(int64_t a,int64_t b)
// uint64_t vrshld_u64(uint64_t a,int64_t b)

TEST_CASE(test_vrshl_s16) {
    struct {
        int16_t a[4];
        int16_t b[4];
        int16_t r[4];
    } test_vec[] = {
        {{-1, 16127, 7767, 21548}, {16, 11, 15, -16}, {0, -2048, INT16_MIN, 0}},
        {{-1, 155, 26190, -23334},
         {-16, 14, 14, -12},
         {0, -16384, INT16_MIN, -6}},
        {{-12578, 28010, -16565, 31821},
         {11, 12, 11, 7269},
         {-4096, -24576, 22528, 0}},
        {{-1065, 21373, 12341, -29402},
         {9, -9, 13, 12},
         {-20992, 42, -24576, 24576}},
        {{-22525, -11632, -2809, -12818},
         {-20684, 11, 10, -16},
         {0, INT16_MIN, 7168, 0}},
        {{14547, -4223, -9107, 25175},
         {14, -15, 14, -11},
         {-16384, 0, 16384, 12}},
        {{-29192, -23726, -20257, 24964},
         {-14, 8, 14, -26376},
         {-2, 20992, -16384, 98}},
        {{-30838, 27999, -18012, -18857},
         {15, -9, 9, 31646},
         {0, 55, 18432, 0}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int16x4_t r = vrshl_s16(a, b);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
