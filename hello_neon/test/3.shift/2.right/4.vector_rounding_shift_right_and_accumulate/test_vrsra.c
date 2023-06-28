// 2023-04-19 18:20
#include <neon.h>
#include <neon_test.h>
// int8x8_t vrsra_n_s8(int8x8_t a,int8x8_t b,const int n)
// int16x4_t vrsra_n_s16(int16x4_t a,int16x4_t b,const int n)
// int32x2_t vrsra_n_s32(int32x2_t a,int32x2_t b,const int n)
// int64x1_t vrsra_n_s64(int64x1_t a,int64x1_t b,const int n)
// uint8x8_t vrsra_n_u8(uint8x8_t a,uint8x8_t b,const int n)
// uint16x4_t vrsra_n_u16(uint16x4_t a,uint16x4_t b,const int n)
// uint32x2_t vrsra_n_u32(uint32x2_t a,uint32x2_t b,const int n)
// uint64x1_t vrsra_n_u64(uint64x1_t a,uint64x1_t b,const int n)
//
// int8x16_t vrsraq_n_s8(int8x16_t a,int8x16_t b,const int n)
// int16x8_t vrsraq_n_s16(int16x8_t a,int16x8_t b,const int n)
// int32x4_t vrsraq_n_s32(int32x4_t a,int32x4_t b,const int n)
// int64x2_t vrsraq_n_s64(int64x2_t a,int64x2_t b,const int n)
// uint8x16_t vrsraq_n_u8(uint8x16_t a,uint8x16_t b,const int n)
// uint16x8_t vrsraq_n_u16(uint16x8_t a,uint16x8_t b,const int n)
// uint32x4_t vrsraq_n_u32(uint32x4_t a,uint32x4_t b,const int n)
// uint64x2_t vrsraq_n_u64(uint64x2_t a,uint64x2_t b,const int n)
// --------------------------------------------------------------
// int64_t vrsrad_n_s64(int64_t a,int64_t b,const int n)
// uint64_t vrsrad_n_u64(uint64_t a,uint64_t b,const int n)

TEST_CASE(test_vrsra_n_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r1[8];
        int8_t r3[8];
        int8_t r5[8];
        int8_t r6[8];
        int8_t r8[8];
    } test_vec[] = {
        {{53, -70, 119, 8, 71, -53, -109, 51},
         {60, -71, -27, -112, -21, 75, 108, -57},
         {83, -105, 106, -48, 61, -15, -55, 23},
         {61, -79, 116, -6, 68, -44, -95, 44},
         {55, -72, 118, 5, 70, -51, -106, 49},
         {54, -71, 119, 6, 71, -52, -107, 50},
         {53, -70, 119, 8, 71, -53, -109, 51}},
        {{-24, 15, 107, 16, 7, -81, -124, -98},
         {26, -104, -113, -127, 68, 60, 111, 121},
         {-11, -37, 51, -47, 41, -51, -68, -37},
         {-21, 2, 93, 0, 16, -73, -110, -83},
         {-23, 12, 103, 12, 9, -79, -121, -94},
         {-24, 13, 105, 14, 8, -80, -122, -96},
         {-24, 15, 107, 16, 7, -81, -124, -98}},
        {{-10, -26, -127, 61, -79, 20, 113, -18},
         {-50, 86, 126, -71, -95, -21, INT8_MIN, -119},
         {-35, 17, -64, 26, -126, 10, 49, -77},
         {-16, -15, -111, 52, -91, 17, 97, -33},
         {-12, -23, -123, 59, -82, 19, 109, -22},
         {-11, -25, -125, 60, -80, 20, 111, -20},
         {-10, -26, -127, 61, -79, 20, 113, -18}},
        {{-6, -21, -102, 2, -101, 30, -96, -75},
         {-73, 47, 55, -5, 107, -90, 116, 98},
         {-42, 3, -74, 0, -47, -15, -38, -26},
         {-15, -15, -95, 1, -88, 19, -81, -63},
         {-8, -20, -100, 2, -98, 27, -92, -72},
         {-7, -20, -101, 2, -99, 29, -94, -73},
         {-6, -21, -102, 2, -101, 30, -96, -75}},
        {{-116, -11, -97, 61, 10, 16, 43, -40},
         {102, -86, -111, 8, -107, 17, -111, -113},
         {-65, -54, 104, 65, -43, 25, -12, -96},
         {-103, -22, -111, 62, -3, 18, 29, -54},
         {-113, -14, -100, 61, 7, 17, 40, -44},
         {-114, -12, -99, 61, 8, 16, 41, -42},
         {-116, -11, -97, 61, 10, 16, 43, -40}},
        {{-3, 43, -111, -104, 74, 49, 77, 1},
         {97, -124, -4, -52, 42, 112, 46, -74},
         {46, -19, -113, 126, 95, 105, 100, -36},
         {9, 28, -111, -110, 79, 63, 83, -8},
         {0, 39, -111, -106, 75, 53, 78, -1},
         {-1, 41, -111, -105, 75, 51, 78, 0},
         {-3, 43, -111, -104, 74, 49, 77, 1}},
        {{101, -50, -12, 111, -34, 31, 71, 69},
         {-55, -40, 77, 94, -22, -34, -18, -25},
         {74, -70, 27, -98, -45, 14, 62, 57},
         {94, -55, -2, 123, -37, 27, 69, 66},
         {99, -51, -10, 114, -35, 30, 70, 68},
         {100, -51, -11, 112, -34, 30, 71, 69},
         {101, -50, -12, 111, -34, 31, 71, 69}},
        {{10, INT8_MAX, INT8_MAX, 84, -79, -52, 85, 18},
         {81, 81, -34, 123, -63, 13, 50, 38},
         {51, -88, 110, -110, -110, -45, 110, 37},
         {20, -119, 123, 99, -87, -50, 91, 23},
         {13, -126, 126, 88, -81, -52, 87, 19},
         {11, INT8_MIN, 126, 86, -80, -52, 86, 19},
         {10, INT8_MAX, INT8_MAX, 84, -79, -52, 85, 18}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);

        int8x8_t r1 = vrsra_n_s8(a, b, 1);
        int8x8_t r3 = vrsra_n_s8(a, b, 3);
        int8x8_t r5 = vrsra_n_s8(a, b, 5);
        int8x8_t r6 = vrsra_n_s8(a, b, 6);
        /* int8x8_t r8 = vrsra_n_s8(a, b, 8); */

        int8x8_t check1 = vld1_s8(test_vec[i].r1);
        int8x8_t check3 = vld1_s8(test_vec[i].r3);
        int8x8_t check5 = vld1_s8(test_vec[i].r5);
        int8x8_t check6 = vld1_s8(test_vec[i].r6);
        /* int8x8_t check8 = vld1_s8(test_vec[i].r8); */

        ASSERT_EQUAL(r1, check1);
        ASSERT_EQUAL(r3, check3);
        ASSERT_EQUAL(r5, check5);
        ASSERT_EQUAL(r6, check6);
        /* ASSERT_EQUAL(r8, check8); */
    }
    return 0;
}
