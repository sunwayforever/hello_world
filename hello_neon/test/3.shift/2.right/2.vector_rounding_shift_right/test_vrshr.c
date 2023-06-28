// 2023-04-19 16:58
#include <neon.h>
#include <neon_test.h>
// int8x8_t vrshr_n_s8(int8x8_t a,const int n)
// int16x4_t vrshr_n_s16(int16x4_t a,const int n)
// int32x2_t vrshr_n_s32(int32x2_t a,const int n)
// int64x1_t vrshr_n_s64(int64x1_t a,const int n)
// uint8x8_t vrshr_n_u8(uint8x8_t a,const int n)
// uint16x4_t vrshr_n_u16(uint16x4_t a,const int n)
// uint32x2_t vrshr_n_u32(uint32x2_t a,const int n)
// uint64x1_t vrshr_n_u64(uint64x1_t a,const int n)
//
// int8x16_t vrshrq_n_s8(int8x16_t a,const int n)
// int16x8_t vrshrq_n_s16(int16x8_t a,const int n)
// int32x4_t vrshrq_n_s32(int32x4_t a,const int n)
// int64x2_t vrshrq_n_s64(int64x2_t a,const int n)
// uint8x16_t vrshrq_n_u8(uint8x16_t a,const int n)
// uint16x8_t vrshrq_n_u16(uint16x8_t a,const int n)
// uint32x4_t vrshrq_n_u32(uint32x4_t a,const int n)
// uint64x2_t vrshrq_n_u64(uint64x2_t a,const int n)
// --------------------------------------------------
// int64_t vrshrd_n_s64(int64_t a,const int n)
// uint64_t vrshrd_n_u64(uint64_t a,const int n)

TEST_CASE(test_vrshr_n_s8) {
    static const struct {
        int8_t a[8];
        int8_t r1[8];
        int8_t r3[8];
        int8_t r5[8];
        int8_t r6[8];
        int8_t r8[8];
    } test_vec[] = {
        {{-87, 13, 107, -109, -49, -33, -55, -61},
         {-43, 7, 54, -54, -24, -16, -27, -30},
         {-11, 2, 13, -14, -6, -4, -7, -8},
         {-3, 0, 3, -3, -2, -1, -2, -2},
         {-1, 0, 2, -2, -1, -1, -1, -1},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{98, -18, -28, 54, -125, 113, 76, -98},
         {49, -9, -14, 27, -62, 57, 38, -49},
         {12, -2, -3, 7, -16, 14, 10, -12},
         {3, -1, -1, 2, -4, 4, 2, -3},
         {2, 0, 0, 1, -2, 2, 1, -2},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{38, 9, -38, 29, 25, -16, -92, -67},
         {19, 5, -19, 15, 13, -8, -46, -33},
         {5, 1, -5, 4, 3, -2, -11, -8},
         {1, 0, -1, 1, 1, 0, -3, -2},
         {1, 0, -1, 0, 0, 0, -1, -1},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{-90, -49, 85, -6, -126, -96, 107, 44},
         {-45, -24, 43, -3, -63, -48, 54, 22},
         {-11, -6, 11, -1, -16, -12, 13, 6},
         {-3, -2, 3, 0, -4, -3, 3, 1},
         {-1, -1, 1, 0, -2, -1, 2, 1},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{-83, -41, -65, 125, -74, -120, 64, 24},
         {-41, -20, -32, 63, -37, -60, 32, 12},
         {-10, -5, -8, 16, -9, -15, 8, 3},
         {-3, -1, -2, 4, -2, -4, 2, 1},
         {-1, -1, -1, 2, -1, -2, 1, 0},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{118, 37, 78, -7, -106, -101, -105, -68},
         {59, 19, 39, -3, -53, -50, -52, -34},
         {15, 5, 10, -1, -13, -13, -13, -8},
         {4, 1, 2, 0, -3, -3, -3, -2},
         {2, 1, 1, 0, -2, -2, -2, -1},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{-92, 113, -39, -66, 97, 125, 123, 7},
         {-46, 57, -19, -33, 49, 63, 62, 4},
         {-11, 14, -5, -8, 12, 16, 15, 1},
         {-3, 4, -1, -2, 3, 4, 4, 0},
         {-1, 2, -1, -1, 2, 2, 2, 0},
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{76, -48, 1, -49, 112, 109, -5, 30},
         {38, -24, 1, -24, 56, 55, -2, 15},
         {10, -6, 0, -6, 14, 14, -1, 4},
         {2, -1, 0, -2, 4, 3, 0, 1},
         {1, -1, 0, -1, 2, 2, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);

        int8x8_t r1 = vrshr_n_s8(a, 1);
        int8x8_t r3 = vrshr_n_s8(a, 3);
        int8x8_t r5 = vrshr_n_s8(a, 5);
        int8x8_t r6 = vrshr_n_s8(a, 6);
        /* int8x8_t r8 = vrshr_n_s8(a, 8); */

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
