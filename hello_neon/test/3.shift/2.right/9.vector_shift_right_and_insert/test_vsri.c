// 2023-04-20 11:23
#include <neon.h>
#include <neon_test.h>
// int8x8_t vsri_n_s8(int8x8_t a,int8x8_t b,const int n)
// int16x4_t vsri_n_s16(int16x4_t a,int16x4_t b,const int n)
// int32x2_t vsri_n_s32(int32x2_t a,int32x2_t b,const int n)
// int64x1_t vsri_n_s64(int64x1_t a,int64x1_t b,const int n)
// uint8x8_t vsri_n_u8(uint8x8_t a,uint8x8_t b,const int n)
// uint16x4_t vsri_n_u16(uint16x4_t a,uint16x4_t b,const int n)
// uint32x2_t vsri_n_u32(uint32x2_t a,uint32x2_t b,const int n)
// uint64x1_t vsri_n_u64(uint64x1_t a,uint64x1_t b,const int n)
//
// int16x8_t vsriq_n_s16(int16x8_t a,int16x8_t b,const int n)
// int8x16_t vsriq_n_s8(int8x16_t a,int8x16_t b,const int n)
// int32x4_t vsriq_n_s32(int32x4_t a,int32x4_t b,const int n)
// int64x2_t vsriq_n_s64(int64x2_t a,int64x2_t b,const int n)
// uint8x16_t vsriq_n_u8(uint8x16_t a,uint8x16_t b,const int n)
// uint16x8_t vsriq_n_u16(uint16x8_t a,uint16x8_t b,const int n)
// uint32x4_t vsriq_n_u32(uint32x4_t a,uint32x4_t b,const int n)
// uint64x2_t vsriq_n_u64(uint64x2_t a,uint64x2_t b,const int n)
// --------------------------------------------------------------
// poly64x1_t vsri_n_p64(poly64x1_t a,poly64x1_t b,const int n)
// poly64x2_t vsriq_n_p64(poly64x2_t a,poly64x2_t b,const int n)
// poly8x8_t vsri_n_p8(poly8x8_t a,poly8x8_t b,const int n)
// poly8x16_t vsriq_n_p8(poly8x16_t a,poly8x16_t b,const int n)
// poly16x4_t vsri_n_p16(poly16x4_t a,poly16x4_t b,const int n)
// poly16x8_t vsriq_n_p16(poly16x8_t a,poly16x8_t b,const int n)
// --------------------------------------------------------------
// int64_t vsrid_n_s64(int64_t a,int64_t b,const int n)
// uint64_t vsrid_n_u64(uint64_t a,uint64_t b,const int n)

TEST_CASE(test_vsri_n_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r1[8];
        int8_t r3[8];
        int8_t r5[8];
        int8_t r8[8];
    } test_vec[] = {
        {{-10, -27, 61, -67, 54, 81, 123, 14},
         {108, -105, -69, -22, 83, 115, -2, -18},
         {-74, -53, 93, -11, 41, 57, INT8_MAX, 119},
         {-19, -14, 55, -67, 42, 78, INT8_MAX, 29},
         {-13, -28, 61, -65, 50, 83, INT8_MAX, 15},
         {-10, -27, 61, -67, 54, 81, 123, 14}},
        {{116, -103, -59, 70, 72, 7, 40, -118},
         {79, 34, 79, -92, -7, -26, 76, -16},
         {39, -111, -89, 82, 124, 115, 38, -8},
         {105, -124, -55, 84, 95, 28, 41, -98},
         {114, -103, -62, 69, 79, 7, 42, -113},
         {116, -103, -59, 70, 72, 7, 40, -118}},
        {{-53, -119, -83, 1, -38, 40, 15, 71},
         {-65, -53, 49, 18, 62, 47, 0, -78},
         {-33, -27, -104, 9, -97, 23, 0, 89},
         {-41, -103, -90, 2, -57, 37, 0, 86},
         {-51, -114, -87, 0, -39, 41, 8, 69},
         {-53, -119, -83, 1, -38, 40, 15, 71}},
        {{-56, -59, -8, 16, -51, 32, -102, 28},
         {67, -23, -64, 60, -49, 12, 44, -102},
         {-95, -12, -32, 30, -25, 6, -106, 77},
         {-56, -35, -8, 7, -39, 33, -123, 19},
         {-54, -57, -2, 17, -50, 32, -103, 28},
         {-56, -59, -8, 16, -51, 32, -102, 28}},
        {{-107, -39, -100, 112, 1, -85, -73, -64},
         {118, -24, -45, -75, 23, -45, 103, -33},
         {-69, -12, -23, 90, 11, -23, -77, -17},
         {-114, -35, -102, 118, 2, -70, -84, -37},
         {-109, -33, -98, 117, 0, -82, -77, -58},
         {-107, -39, -100, 112, 1, -85, -73, -64}},
        {{-103, 96, -17, 102, INT8_MIN, -119, -126, -61},
         {115, 66, 0, 66, 78, 44, -35, -29},
         {-71, 33, INT8_MIN, 33, -89, -106, -18, -15},
         {-114, 104, -32, 104, -119, -123, -101, -36},
         {-101, 98, -24, 98, -126, -119, -122, -57},
         {-103, 96, -17, 102, INT8_MIN, -119, -126, -61}},
        {{6, 121, 83, 7, 36, 10, -56, -101},
         {-14, -101, 80, 9, 110, -73, -24, 7},
         {121, 77, 40, 4, 55, 91, -12, -125},
         {30, 115, 74, 1, 45, 22, -35, INT8_MIN},
         {7, 124, 82, 0, 35, 13, -49, -104},
         {6, 121, 83, 7, 36, 10, -56, -101}},
        {{23, -41, 109, -104, 97, -17, 91, -44},
         {49, 91, 22, INT8_MAX, -120, -13, 99, -114},
         {24, -83, 11, -65, 68, -7, 49, -57},
         {6, -53, 98, -113, 113, -2, 76, -47},
         {17, -46, 104, -101, 100, -17, 91, -44},
         {23, -41, 109, -104, 97, -17, 91, -44}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);

        int8x8_t r1 = vsri_n_s8(a, b, 1);
        int8x8_t r3 = vsri_n_s8(a, b, 3);
        int8x8_t r5 = vsri_n_s8(a, b, 5);
        /* int8x8_t r8 = vsri_n_s8(a, b, 8); */

        int8x8_t check1 = vld1_s8(test_vec[i].r1);
        int8x8_t check3 = vld1_s8(test_vec[i].r3);
        int8x8_t check5 = vld1_s8(test_vec[i].r5);
        /* int8x8_t check8 = vld1_s8(test_vec[i].r8); */

        ASSERT_EQUAL(r1, check1);
        ASSERT_EQUAL(r3, check3);
        ASSERT_EQUAL(r5, check5);
        /* ASSERT_EQUAL(r8, check8); */
    }
    return 0;
}
