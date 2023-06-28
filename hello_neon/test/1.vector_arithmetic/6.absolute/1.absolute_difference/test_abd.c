// 2023-04-17 15:35
#include <neon.h>
#include <neon_test.h>

// int8x8_t vabd_s8(int8x8_t a,int8x8_t b)
// int16x4_t vabd_s16(int16x4_t a,int16x4_t b)
// int32x2_t vabd_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vabd_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vabd_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vabd_u32(uint32x2_t a,uint32x2_t b)
//
// int8x16_t vabdq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vabdq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vabdq_s32(int32x4_t a,int32x4_t b)
// uint8x16_t vabdq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vabdq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vabdq_u32(uint32x4_t a,uint32x4_t b)
// ----------------------------------------------
// float32x2_t vabd_f32(float32x2_t a,float32x2_t b)
// float64x1_t vabd_f64(float64x1_t a,float64x1_t b)
//
// float32x4_t vabdq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vabdq_f64(float64x2_t a,float64x2_t b)
// ----------------------------------------------
// float32_t vabds_f32(float32_t a,float32_t b)
// float64_t vabdd_f64(float64_t a,float64_t b)

TEST_CASE(test_vabd_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{INT8_MAX, INT8_MAX, 10, 10, 103, 22, -16, -30},
         {-1, -2, 20, -10, -111, 6, -68, 126},
         {INT8_MIN, -127, 10, 20, -42, 16, 52, -100}},
        {{18, 60, 4, 117, 103, 22, -16, -30},
         {87, -83, -104, -48, -111, 6, -68, 126},
         {69, -113, 108, -91, -42, 16, 52, -100}},
        {{-87, 114, 114, 23, -5, 68, -72, -50},
         {90, -66, -84, 51, -90, -91, 49, -72},
         {-79, -76, -58, 28, 85, -97, 121, 22}},
        {{-31, 53, 45, 72, 75, 29, 42, -94},
         {-53, -62, 115, 92, -55, 47, -38, 114},
         {22, 115, 70, 20, -126, 18, 80, -48}},
        {{-94, 77, -118, -99, -111, 66, 107, -21},
         {0, 24, 30, -90, -67, 79, 95, -97},
         {94, 53, -108, 9, 44, 13, 12, 76}},
        {{-123, -116, -25, -48, -86, 18, 115, 117},
         {-44, -26, -47, -99, 21, -85, 16, -73},
         {79, 90, 22, 51, 107, 103, 99, -66}},
        {{-8, -102, 84, -119, -36, -64, 116, -35},
         {-40, -109, -125, -107, -30, -30, 52, 103},
         {32, 7, -47, 12, 6, 34, 64, -118}},
        {{111, 28, 56, 25, 46, -85, -114, 2},
         {-111, 95, -96, -90, 10, -80, 94, 3},
         {-34, 67, -104, 115, 36, 5, -48, 1}},
        {{74, -78, -116, 38, 114, 1, 3, 74},
         {-108, -121, -32, 118, 105, 20, -34, -40},
         {-74, 43, 84, 80, 9, 19, 37, 114}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vabd_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
