// 2023-04-21 16:06
#include <neon.h>
#include <neon_test.h>
// int8x8_t vuzp1_s8(int8x8_t a,int8x8_t b)
// int16x4_t vuzp1_s16(int16x4_t a,int16x4_t b)
// int32x2_t vuzp1_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vuzp1_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vuzp1_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vuzp1_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vuzp1_f32(float32x2_t a,float32x2_t b)
// poly8x8_t vuzp1_p8(poly8x8_t a,poly8x8_t b)
// poly16x4_t vuzp1_p16(poly16x4_t a,poly16x4_t b)
//
// int8x16_t vuzp1q_s8(int8x16_t a,int8x16_t b)
// int16x8_t vuzp1q_s16(int16x8_t a,int16x8_t b)
// int32x4_t vuzp1q_s32(int32x4_t a,int32x4_t b)
// int64x2_t vuzp1q_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vuzp1q_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vuzp1q_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vuzp1q_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vuzp1q_u64(uint64x2_t a,uint64x2_t b)
// poly64x2_t vuzp1q_p64(poly64x2_t a,poly64x2_t b)
// float32x4_t vuzp1q_f32(float32x4_t a,float32x4_t b)
// float64x2_t vuzp1q_f64(float64x2_t a,float64x2_t b)
// poly8x16_t vuzp1q_p8(poly8x16_t a,poly8x16_t b)
// poly16x8_t vuzp1q_p16(poly16x8_t a,poly16x8_t b)
// ----------------------------------------------------
// int8x8_t vuzp2_s8(int8x8_t a,int8x8_t b)
// int16x4_t vuzp2_s16(int16x4_t a,int16x4_t b)
// int32x2_t vuzp2_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vuzp2_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vuzp2_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vuzp2_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vuzp2_f32(float32x2_t a,float32x2_t b)
// poly8x8_t vuzp2_p8(poly8x8_t a,poly8x8_t b)
// poly16x4_t vuzp2_p16(poly16x4_t a,poly16x4_t b)
// int8x8x2_t vuzp_s8(int8x8_t a,int8x8_t b)
// int16x4x2_t vuzp_s16(int16x4_t a,int16x4_t b)
// int32x2x2_t vuzp_s32(int32x2_t a,int32x2_t b)
// float32x2x2_t vuzp_f32(float32x2_t a,float32x2_t b)
// uint8x8x2_t vuzp_u8(uint8x8_t a,uint8x8_t b)
// uint16x4x2_t vuzp_u16(uint16x4_t a,uint16x4_t b)
// uint32x2x2_t vuzp_u32(uint32x2_t a,uint32x2_t b)
// poly8x8x2_t vuzp_p8(poly8x8_t a,poly8x8_t b)
// poly16x4x2_t vuzp_p16(poly16x4_t a,poly16x4_t b)
//
// int8x16_t vuzp2q_s8(int8x16_t a,int8x16_t b)
// int16x8_t vuzp2q_s16(int16x8_t a,int16x8_t b)
// int32x4_t vuzp2q_s32(int32x4_t a,int32x4_t b)
// int64x2_t vuzp2q_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vuzp2q_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vuzp2q_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vuzp2q_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vuzp2q_u64(uint64x2_t a,uint64x2_t b)
// poly64x2_t vuzp2q_p64(poly64x2_t a,poly64x2_t b)
// float32x4_t vuzp2q_f32(float32x4_t a,float32x4_t b)
// float64x2_t vuzp2q_f64(float64x2_t a,float64x2_t b)
// poly8x16_t vuzp2q_p8(poly8x16_t a,poly8x16_t b)
// poly16x8_t vuzp2q_p16(poly16x8_t a,poly16x8_t b)
// int8x16x2_t vuzpq_s8(int8x16_t a,int8x16_t b)
// int16x8x2_t vuzpq_s16(int16x8_t a,int16x8_t b)
// int32x4x2_t vuzpq_s32(int32x4_t a,int32x4_t b)
// float32x4x2_t vuzpq_f32(float32x4_t a,float32x4_t b)
// uint8x16x2_t vuzpq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8x2_t vuzpq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4x2_t vuzpq_u32(uint32x4_t a,uint32x4_t b)
// poly8x16x2_t vuzpq_p8(poly8x16_t a,poly8x16_t b)
// poly16x8x2_t vuzpq_p16(poly16x8_t a,poly16x8_t b)

TEST_CASE(test_vuzp1_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-118, -36, 91, -109, -14, 74, -81, 58},
         {-10, INT8_MIN, -52, -53, -80, 35, -119, 69},
         {-118, 91, -14, -81, -10, -52, -80, -119}},
        {{-37, 112, -36, 99, 49, -105, 96, 97},
         {79, -113, -17, 16, -104, 44, -121, 35},
         {-37, -36, 49, 96, 79, -17, -104, -121}},
        {{8, -30, -74, -5, 45, 101, 53, 35},
         {-27, 1, -18, -107, 36, 119, -37, -1},
         {8, -74, 45, 53, -27, -18, 36, -37}},
        {{-25, -73, 99, 25, 79, -61, 122, -98},
         {83, 106, -82, -21, -106, 53, 14, -98},
         {-25, 99, 79, 122, 83, -82, -106, 14}},
        {{24, -60, -103, 69, 41, -50, 104, 15},
         {-49, 86, -92, -12, -50, INT8_MAX, -13, -75},
         {24, -103, 41, 104, -49, -92, -50, -13}},
        {{55, 86, -50, -122, 26, 73, 36, 109},
         {-77, -46, 88, 73, 7, 103, -25, 31},
         {55, -50, 26, 36, -77, 88, 7, -25}},
        {{43, -127, 100, 85, 79, -52, 100, 31},
         {35, 8, 19, -15, -120, 6, -90, -65},
         {43, 100, 79, 100, 35, 19, -120, -90}},
        {{93, 117, 69, 119, -66, 105, -28, 113},
         {59, 60, -70, 66, -93, -95, 98, -49},
         {93, 69, -66, -28, 59, -70, -93, 98}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vuzp1_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
