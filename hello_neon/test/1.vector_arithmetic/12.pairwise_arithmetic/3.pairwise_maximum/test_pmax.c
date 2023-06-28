// 2023-04-18 14:05
#include <neon.h>
#include <neon_test.h>
// int8x8_t vpmax_s8(int8x8_t a,int8x8_t b)
// int16x4_t vpmax_s16(int16x4_t a,int16x4_t b)
// int32x2_t vpmax_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vpmax_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vpmax_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vpmax_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vpmax_f32(float32x2_t a,float32x2_t b)
//
// int8x16_t vpmaxq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vpmaxq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vpmaxq_s32(int32x4_t a,int32x4_t b)
// uint8x16_t vpmaxq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vpmaxq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vpmaxq_u32(uint32x4_t a,uint32x4_t b)
// float32x4_t vpmaxq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vpmaxq_f64(float64x2_t a,float64x2_t b)
// ---------------------------------------------------
// float32_t vpmaxs_f32(float32x2_t a)
// float64_t vpmaxqd_f64(float64x2_t a)
// float32_t vpmaxnms_f32(float32x2_t a)
//                ^^---与 vmaxnm 相同, nm 表示 max(nan, xxx) 返回 xxx
// float64_t vpmaxnmqd_f64(float64x2_t a)

TEST_CASE(test_vpmax_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{54, 72, 21, 4, -44, -85, -47, 48},
         {-97, 124, -89, -115, 72, -78, 21, 9},
         {72, 21, -44, 48, 124, -89, 72, 21}},
        {{47, 79, 36, -113, 55, -81, 42, 63},
         {81, -48, 68, -24, -31, -126, -127, 23},
         {79, 36, 55, 63, 81, 68, -31, 23}},
        {{-54, -106, 28, -98, 66, -19, -49, -31},
         {105, 118, 111, -78, 41, -124, -69, 88},
         {-54, 28, 66, -31, 118, 111, 41, 88}},
        {{-45, -33, -25, 10, -113, 17, 73, -32},
         {-31, -115, -56, -62, 15, 73, -38, -38},
         {-33, 10, 17, 73, -31, -56, 73, -38}},
        {{-33, -10, 120, 33, -29, 71, 3, 76},
         {-66, 114, -2, -25, -10, -70, 63, -55},
         {-10, 120, 71, 76, 114, -2, -10, 63}},
        {{-103, 38, -45, 40, 55, 28, 8, 24},
         {-87, -48, -37, -71, 25, -75, -109, -7},
         {38, 40, 55, 24, -48, -37, 25, -7}},
        {{-85, 11, 26, -114, 83, 29, -38, 17},
         {-113, -39, -8, -123, -109, 55, 78, 44},
         {11, 26, 83, 17, -39, -8, 55, 78}},
        {{93, 33, 85, -108, 61, 93, -84, -25},
         {46, -121, -96, 71, 60, 51, 64, -25},
         {93, 85, 93, -25, 46, 71, 60, 64}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vpmax_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }

    return 0;
}
