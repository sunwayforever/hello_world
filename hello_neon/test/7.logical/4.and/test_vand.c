// 2023-04-20 17:08
#include <neon.h>
#include <neon_test.h>
// int8x8_t vand_s8(int8x8_t a,int8x8_t b)
// int16x4_t vand_s16(int16x4_t a,int16x4_t b)
// int32x2_t vand_s32(int32x2_t a,int32x2_t b)
// int64x1_t vand_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vand_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vand_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vand_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vand_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t vandq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vandq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vandq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vandq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vandq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vandq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vandq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vandq_u64(uint64x2_t a,uint64x2_t b)

TEST_CASE(test_vand_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{20, -27, -113, 29, 103, 9, -114, 56},
         {-67, 79, 119, -97, 47, 81, -48, 83},
         {20, 69, 7, 29, 39, 1, INT8_MIN, 16}},
        {{49, 9, -33, 41, -35, 105, -38, 92},
         {-11, -35, -64, -71, 47, -42, -74, 67},
         {49, 9, -64, 41, 13, 64, -110, 64}},
        {{-69, 69, 96, 34, 78, -18, 90, 11},
         {61, -46, -86, 108, 35, 123, -64, 84},
         {57, 64, 32, 32, 2, 106, 64, 0}},
        {{-124, -97, 126, 97, 8, 88, -67, -3},
         {53, 125, -73, 101, 83, 109, -88, 15},
         {4, 29, 54, 97, 0, 72, -88, 13}},
        {{-78, 9, 49, 1, -9, -116, 12, 53},
         {94, -73, -95, -127, 50, 97, -42, -74},
         {18, 1, 33, 1, 50, 0, 4, 52}},
        {{1, 84, 23, 9, -84, -44, 7, -31},
         {82, -66, 70, -91, 43, -17, -76, -35},
         {0, 20, 6, 1, 40, -60, 4, -63}},
        {{-8, -26, -34, -17, 114, -21, 36, -48},
         {-94, -58, 81, -44, 39, 39, -118, 40},
         {-96, -58, 80, -60, 34, 35, 0, 0}},
        {{123, -95, 50, 39, 117, 57, 9, -57},
         {-9, 79, 109, 34, 62, 33, -1, 54},
         {115, 1, 32, 34, 52, 33, 9, 6}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vand_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
