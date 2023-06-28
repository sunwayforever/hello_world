// 2023-04-15 21:53
#include <neon.h>
#include <neon_test.h>
// int8x8_t vmla_s8(int8x8_t a,int8x8_t b,int8x8_t c)
//           ^^^--- multiply accumulate, r[i]=a[i]+b[i]*c[i]
// int16x4_t vmla_s16(int16x4_t a,int16x4_t b,int16x4_t c)
// int32x2_t vmla_s32(int32x2_t a,int32x2_t b,int32x2_t c)
// uint8x8_t vmla_u8(uint8x8_t a,uint8x8_t b,uint8x8_t c)
// uint16x4_t vmla_u16(uint16x4_t a,uint16x4_t b,uint16x4_t c)
// uint32x2_t vmla_u32(uint32x2_t a,uint32x2_t b,uint32x2_t c)
//
// int8x16_t vmlaq_s8(int8x16_t a,int8x16_t b,int8x16_t c)
//               ^--- 128-bit vector
// int16x8_t vmlaq_s16(int16x8_t a,int16x8_t b,int16x8_t c)
// int32x4_t vmlaq_s32(int32x4_t a,int32x4_t b,int32x4_t c)
// uint8x16_t vmlaq_u8(uint8x16_t a,uint8x16_t b,uint8x16_t c)
// uint16x8_t vmlaq_u16(uint16x8_t a,uint16x8_t b,uint16x8_t c)
// uint32x4_t vmlaq_u32(uint32x4_t a,uint32x4_t b,uint32x4_t c)
// --------------------------------------------------------------
// float32x2_t vmla_f32(float32x2_t a,float32x2_t b,float32x2_t c)
// float64x1_t vmla_f64(float64x1_t a,float64x1_t b,float64x1_t c)
//
// float32x4_t vmlaq_f32(float32x4_t a,float32x4_t b,float32x4_t c)
// float64x2_t vmlaq_f64(float64x2_t a,float64x2_t b,float64x2_t c)
// --------------------------------------------------------------
// int8x8_t vmls_s8(int8x8_t a,int8x8_t b,int8x8_t c)
//           ^^^--- multiple subtract, r[i]=a[i]-b[i]*c[i]
// int16x4_t vmls_s16(int16x4_t a,int16x4_t b,int16x4_t c)
// int32x2_t vmls_s32(int32x2_t a,int32x2_t b,int32x2_t c)
// uint8x8_t vmls_u8(uint8x8_t a,uint8x8_t b,uint8x8_t c)
// uint16x4_t vmls_u16(uint16x4_t a,uint16x4_t b,uint16x4_t c)
// uint32x2_t vmls_u32(uint32x2_t a,uint32x2_t b,uint32x2_t c)
//
// int8x16_t vmlsq_s8(int8x16_t a,int8x16_t b,int8x16_t c)
// int16x8_t vmlsq_s16(int16x8_t a,int16x8_t b,int16x8_t c)
// int32x4_t vmlsq_s32(int32x4_t a,int32x4_t b,int32x4_t c)
// uint8x16_t vmlsq_u8(uint8x16_t a,uint8x16_t b,uint8x16_t c)
// uint16x8_t vmlsq_u16(uint16x8_t a,uint16x8_t b,uint16x8_t c)
// uint32x4_t vmlsq_u32(uint32x4_t a,uint32x4_t b,uint32x4_t c)
// --------------------------------------------------------------
// float32x2_t vmls_f32(float32x2_t a,float32x2_t b,float32x2_t c)
// float64x1_t vmls_f64(float64x1_t a,float64x1_t b,float64x1_t c)
//
// float32x4_t vmlsq_f32(float32x4_t a,float32x4_t b,float32x4_t c)
// float64x2_t vmlsq_f64(float64x2_t a,float64x2_t b,float64x2_t c)
//
TEST_CASE(test_vmla_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t c[8];
        int8_t r[8];
    } test_vec[] = {
        {{-97, -50, -94, -28, 126, -17, -96, 91},
         {-103, -58, 111, -40, 71, 77, 29, -87},
         {-124, 119, -122, 94, -88, 3, -29, -35},
         {-125, -40, -68, 52, 22, -42, 87, 64}},
        {{18, 99, 76, 75, 95, -107, 108, -2},
         {100, 14, -30, -30, -3, -125, 61, -105},
         {73, -84, 111, -112, -6, -116, 58, 126},
         {-106, -53, 74, 107, 113, 57, 62, 80}},
        {{4, -64, -36, -84, -61, -65, -119, -42},
         {35, -43, 33, -126, 107, -114, INT8_MIN, -49},
         {-100, 99, -79, -102, -26, -17, 49, 47},
         {88, 31, -83, -32, -27, 81, 9, -41}},
        {{-101, -96, -65, -107, 45, -7, 19, 49},
         {-70, -16, -35, 125, -81, 102, 83, -46},
         {59, 117, 84, -90, 3, -43, 117, -97},
         {121, 80, 67, -93, 58, -41, 2, -97}},
        {{56, 39, 57, 30, 22, 106, 77, -79},
         {11, 12, 71, 56, 6, 90, 105, -64},
         {74, 70, 61, -6, -84, -111, -52, -25},
         {102, 111, 36, -50, 30, 100, -7, -15}},
        {{6, 33, -114, 9, -10, 3, -88, 46},
         {42, -30, 76, 64, 76, -103, -14, 87},
         {-91, 57, -113, -85, -109, -8, 107, -34},
         {24, 115, 2, -55, -102, 59, -50, -96}},
        {{62, -87, -40, -22, 58, -92, -46, 64},
         {-59, 96, 73, -69, 99, -15, -23, -114},
         {-45, 53, -50, 32, -50, -64, 119, 116},
         {-99, -119, -106, 74, -28, 100, 33, -104}},
        {{-7, 7, 31, -115, -1, -117, 107, 62},
         {52, 67, 40, 110, -25, -6, -82, -83},
         {90, -9, 104, -66, -24, 82, 76, -68},
         {65, -84, 95, 49, 87, -97, 19, 74}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t c = vld1_s8(test_vec[i].c);
        int8x8_t r = vmla_s8(a, b, c);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vmls_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t c[8];
        int8_t r[8];
    } test_vec[] = {
        {{-109, 57, -86, 50, -11, 88, -121, -63},
         {31, -117, 39, 85, -105, -3, 78, -42},
         {53, 25, 95, -29, 65, 107, -31, 54},
         {40, -90, 49, -45, -98, -103, -7, -99}},
        {{-93, -72, -124, 46, 126, 99, 61, 18},
         {-100, -24, 68, -111, 64, -53, 83, 95},
         {86, 122, -76, -19, 119, 2, -61, -84},
         {59, 40, -76, -15, -66, -51, 4, 62}},
        {{28, 34, -113, 93, -115, 112, -108, 49},
         {40, 24, 95, -90, 123, -99, -72, 24},
         {-123, -4, -87, -59, -57, -4, 36, 29},
         {84, -126, -40, -97, -16, -28, -76, 121}},
        {{118, -40, 11, -19, -37, -50, -103, -9},
         {-15, 40, 84, 126, -104, -24, -81, -64},
         {1, 15, 103, 124, -84, 31, -108, 49},
         {-123, INT8_MIN, 63, -27, -69, -74, 109, 55}},
        {{28, 62, -10, -29, 58, 26, 1, -79},
         {-14, 12, -98, -51, -38, 56, -60, -53},
         {96, 25, 74, -7, 1, -7, -71, 2},
         {92, 18, 74, 126, 96, -94, 93, 27}},
        {{8, 32, INT8_MAX, -76, 64, 19, -27, 92},
         {81, -37, 63, -116, -11, 64, 61, -24},
         {76, -37, -75, 39, 19, 122, -14, 116},
         {-4, -57, -12, 96, 17, -109, 59, 60}},
        {{-109, 60, 109, -108, 54, 38, -105, 62},
         {71, 22, -13, -121, 41, -40, -29, 123},
         {-76, 34, 7, -87, 99, 68, -111, -81},
         {-89, 80, -56, 117, 91, -58, 4, 41}},
        {{31, 71, -42, 51, -63, -55, -89, 84},
         {5, 20, -24, 59, 58, INT8_MAX, 122, -127},
         {-107, 109, 8, -65, 69, -21, 58, -7},
         {54, -61, -106, 46, 31, 52, 3, -37}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t c = vld1_s8(test_vec[i].c);
        int8x8_t r = vmls_s8(a, b, c);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
