// 2023-04-20 17:16
#include <neon.h>
#include <neon_test.h>
// int8x8_t vorr_s8(int8x8_t a,int8x8_t b)
// int16x4_t vorr_s16(int16x4_t a,int16x4_t b)
// int32x2_t vorr_s32(int32x2_t a,int32x2_t b)
// int64x1_t vorr_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vorr_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vorr_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vorr_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vorr_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t vorrq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vorrq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vorrq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vorrq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vorrq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vorrq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vorrq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vorrq_u64(uint64x2_t a,uint64x2_t b)

TEST_CASE(test_vorr_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{43, -72, 13, -58, -99, 110, -120, -100},
         {21, -25, -50, 76, 92, 65, 115, -62},
         {63, -1, -49, -50, -35, 111, -5, -34}},
        {{30, -126, 57, 41, 64, 98, -52, -70},
         {41, 28, INT8_MAX, 96, -11, 40, 71, 32},
         {63, -98, INT8_MAX, 105, -11, 106, -49, -70}},
        {{-32, 84, -25, 125, -61, 111, 25, -40},
         {87, -25, 36, -77, 40, -105, 117, 71},
         {-9, -9, -25, -1, -21, -1, 125, -33}},
        {{26, -82, 112, 90, 17, 60, 20, 58},
         {88, -108, -102, 77, -68, -31, 109, -100},
         {90, -66, -6, 95, -67, -3, 125, -66}},
        {{54, 84, 25, -7, -60, 50, -47, 27},
         {25, -10, -50, 65, -115, 67, -120, -89},
         {63, -10, -33, -7, -51, 115, -39, -65}},
        {{-15, -8, 1, 2, 52, 22, 60, -116},
         {-86, -41, -39, 102, -72, 71, 2, -18},
         {-5, -1, -39, 102, -68, 87, 62, -18}},
        {{-101, 27, -25, 95, 77, -71, 122, 102},
         {-81, 72, -89, 60, -117, 48, -28, 125},
         {-65, 91, -25, INT8_MAX, -49, -71, -2, INT8_MAX}},
        {{40, -27, INT8_MAX, 93, -5, -68, -23, -91},
         {-109, -61, 11, 75, 10, 13, 58, -91},
         {-69, -25, INT8_MAX, 95, -5, -67, -5, -91}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vorr_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
