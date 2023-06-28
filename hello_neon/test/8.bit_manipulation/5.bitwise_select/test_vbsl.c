// 2023-04-20 18:52
#include <neon.h>
#include <neon_test.h>
// int8x8_t vbsl_s8(uint8x8_t a,int8x8_t b,int8x8_t c)
// int16x4_t vbsl_s16(uint16x4_t a,int16x4_t b,int16x4_t c)
// int32x2_t vbsl_s32(uint32x2_t a,int32x2_t b,int32x2_t c)
// int64x1_t vbsl_s64(uint64x1_t a,int64x1_t b,int64x1_t c)
// uint8x8_t vbsl_u8(uint8x8_t a,uint8x8_t b,uint8x8_t c)
// uint16x4_t vbsl_u16(uint16x4_t a,uint16x4_t b,uint16x4_t c)
// uint32x2_t vbsl_u32(uint32x2_t a,uint32x2_t b,uint32x2_t c)
// uint64x1_t vbsl_u64(uint64x1_t a,uint64x1_t b,uint64x1_t c)
// poly64x1_t vbsl_p64(poly64x1_t a,poly64x1_t b,poly64x1_t c)
// float32x2_t vbsl_f32(uint32x2_t a,float32x2_t b,float32x2_t c)
// poly8x8_t vbsl_p8(uint8x8_t a,poly8x8_t b,poly8x8_t c)
// poly16x4_t vbsl_p16(uint16x4_t a,poly16x4_t b,poly16x4_t c)
// float64x1_t vbsl_f64(uint64x1_t a,float64x1_t b,float64x1_t c)
//
// int8x16_t vbslq_s8(uint8x16_t a,int8x16_t b,int8x16_t c)
// int16x8_t vbslq_s16(uint16x8_t a,int16x8_t b,int16x8_t c)
// int32x4_t vbslq_s32(uint32x4_t a,int32x4_t b,int32x4_t c)
// int64x2_t vbslq_s64(uint64x2_t a,int64x2_t b,int64x2_t c)
// uint8x16_t vbslq_u8(uint8x16_t a,uint8x16_t b,uint8x16_t c)
// uint16x8_t vbslq_u16(uint16x8_t a,uint16x8_t b,uint16x8_t c)
// uint32x4_t vbslq_u32(uint32x4_t a,uint32x4_t b,uint32x4_t c)
// uint64x2_t vbslq_u64(uint64x2_t a,uint64x2_t b,uint64x2_t c)
// poly64x2_t vbslq_p64(poly64x2_t a,poly64x2_t b,poly64x2_t c)
// float32x4_t vbslq_f32(uint32x4_t a,float32x4_t b,float32x4_t c)
// poly8x16_t vbslq_p8(uint8x16_t a,poly8x16_t b,poly8x16_t c)
// poly16x8_t vbslq_p16(uint16x8_t a,poly16x8_t b,poly16x8_t c)
// float64x2_t vbslq_f64(uint64x2_t a,float64x2_t b,float64x2_t c)

TEST_CASE(test_vbsl_s8) {
    struct {
        uint8_t a[8];
        int8_t b[8];
        int8_t c[8];
        int8_t r[8];
    } test_vec[] = {
        {{121, 28, 45, 151, 120, 7, 82, 78},
         {121, -36, -63, 101, 97, -69, 84, 14},
         {-55, 67, -91, 27, -125, -19, -21, -34},
         {-7, 95, -127, 13, -29, -21, -7, -98}},
        {{174, 64, 203, 94, 160, 0, 116, 25},
         {28, -95, -79, -108, -88, 3, -30, 34},
         {-33, -92, -121, 64, 95, -37, 79, 40},
         {93, -92, -123, 20, -1, -37, 107, 32}},
        {{30, 244, 67, 161, 226, 47, 127, 144},
         {111, 74, -17, 15, 75, 99, 41, 103},
         {5, -38, -4, -83, -35, -34, -49, -68},
         {15, 74, -1, 13, 95, -13, -87, 44}},
        {{130, 86, 252, 225, 49, 75, 10, 79},
         {64, 77, -15, 34, 124, 112, -78, -20},
         {-69, -95, -5, 6, 5, 36, 109, 10},
         {57, -27, -13, 38, 52, 100, 103, 76}},
        {{254, 105, 183, 219, 72, 135, 151, 202},
         {-35, -108, -84, 15, -33, -74, 94, 31},
         {3, 79, 65, INT8_MIN, -64, -12, 108, 123},
         {-35, 6, -28, 11, -56, -10, 126, 59}},
        {{149, 103, 129, 154, 140, 238, 164, 138},
         {88, 92, 102, -96, -29, -3, 106, -64},
         {-111, 22, -49, 113, -52, 46, -112, -48},
         {16, 84, 78, -31, -64, -20, 48, -48}},
        {{125, 210, 80, 61, 198, 188, 184, 91},
         {35, 57, -10, -81, 40, -102, 58, INT8_MIN},
         {-10, -96, 32, -39, -99, -118, -102, 47},
         {-93, 48, 112, -19, 25, -102, 58, 36}},
        {{161, 105, 160, 109, 151, 48, 61, 21},
         {2, -115, 82, -56, 73, 11, 36, 109},
         {68, 26, 28, 108, -76, 86, -20, -85},
         {68, 27, 28, 72, 33, 70, -28, -81}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint8x8_t a = vld1_u8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t c = vld1_s8(test_vec[i].c);
        int8x8_t r = vbsl_s8(a, b, c);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
