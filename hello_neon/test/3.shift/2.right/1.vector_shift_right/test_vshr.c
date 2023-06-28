// 2023-04-19 16:16
#include <neon.h>
#include <neon_test.h>
// int8x8_t vshr_n_s8(int8x8_t a,const int n)
// int16x4_t vshr_n_s16(int16x4_t a,const int n)
// int32x2_t vshr_n_s32(int32x2_t a,const int n)
// int64x1_t vshr_n_s64(int64x1_t a,const int n)
// uint8x8_t vshr_n_u8(uint8x8_t a,const int n)
// uint16x4_t vshr_n_u16(uint16x4_t a,const int n)
// uint32x2_t vshr_n_u32(uint32x2_t a,const int n)
// uint64x1_t vshr_n_u64(uint64x1_t a,const int n)
//
// int8x16_t vshrq_n_s8(int8x16_t a,const int n)
// int16x8_t vshrq_n_s16(int16x8_t a,const int n)
// int32x4_t vshrq_n_s32(int32x4_t a,const int n)
// int64x2_t vshrq_n_s64(int64x2_t a,const int n)
// uint8x16_t vshrq_n_u8(uint8x16_t a,const int n)
// uint16x8_t vshrq_n_u16(uint16x8_t a,const int n)
// uint32x4_t vshrq_n_u32(uint32x4_t a,const int n)
// uint64x2_t vshrq_n_u64(uint64x2_t a,const int n)
// ------------------------------------------------
// int64_t vshrd_n_s64(int64_t a,const int n)
// uint64_t vshrd_n_u64(uint64_t a,const int n)

TEST_CASE(test_vshr_n_s16) {
    struct {
        int16_t a[4];
        int16_t r3[4];
        int16_t r6[4];
        int16_t r10[4];
        int16_t r13[4];
        int16_t r16[4];
    } test_vec[] = {
        {{2391, -30287, 21648, -9648},
         {298, -3786, 2706, -1206},
         {37, -474, 338, -151},
         {2, -30, 21, -10},
         {0, -4, 2, -2},
         {0, -1, 0, -1}},
        {{-32696, -7749, 3517, -3032},
         {-4087, -969, 439, -379},
         {-511, -122, 54, -48},
         {-32, -8, 3, -3},
         {-4, -1, 0, -1},
         {-1, -1, 0, -1}},
        {{-25896, -19991, -18945, 17860},
         {-3237, -2499, -2369, 2232},
         {-405, -313, -297, 279},
         {-26, -20, -19, 17},
         {-4, -3, -3, 2},
         {-1, -1, -1, 0}},
        {{-15514, -7567, -4534, -24034},
         {-1940, -946, -567, -3005},
         {-243, -119, -71, -376},
         {-16, -8, -5, -24},
         {-2, -1, -1, -3},
         {-1, -1, -1, -1}},
        {{-12297, -30933, 31524, 27746},
         {-1538, -3867, 3940, 3468},
         {-193, -484, 492, 433},
         {-13, -31, 30, 27},
         {-2, -4, 3, 3},
         {-1, -1, 0, 0}},
        {{7675, -18355, 30250, 940},
         {959, -2295, 3781, 117},
         {119, -287, 472, 14},
         {7, -18, 29, 0},
         {0, -3, 3, 0},
         {0, -1, 0, 0}},
        {{-27376, 4020, 30795, -20139},
         {-3422, 502, 3849, -2518},
         {-428, 62, 481, -315},
         {-27, 3, 30, -20},
         {-4, 0, 3, -3},
         {-1, 0, 0, -1}},
        {{-14789, -31085, -20044, -21720},
         {-1849, -3886, -2506, -2715},
         {-232, -486, -314, -340},
         {-15, -31, -20, -22},
         {-2, -4, -3, -3},
         {-1, -1, -1, -1}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);

        int16x4_t r3 = vshr_n_s16(a, 3);
        int16x4_t r6 = vshr_n_s16(a, 6);
        int16x4_t r10 = vshr_n_s16(a, 10);
        int16x4_t r13 = vshr_n_s16(a, 13);
        /* int16x4_t r16 = vshr_n_s16(a, 16); */

        int16x4_t check3 = vld1_s16(test_vec[i].r3);
        int16x4_t check6 = vld1_s16(test_vec[i].r6);
        int16x4_t check10 = vld1_s16(test_vec[i].r10);
        int16x4_t check13 = vld1_s16(test_vec[i].r13);
        /* int16x4_t check16 = vld1_s16(test_vec[i].r16); */

        ASSERT_EQUAL(r3, check3);
        ASSERT_EQUAL(r6, check6);
        ASSERT_EQUAL(r10, check10);
        ASSERT_EQUAL(r13, check13);
        /* ASSERT_EQUAL(r16, check16); */
    }
    return 0;
}

TEST_CASE(test_vshr_n_u16) {
    struct {
        uint16_t a[4];
        uint16_t r3[4];
        uint16_t r6[4];
        uint16_t r10[4];
        uint16_t r13[4];
        uint16_t r16[4];
    } test_vec[] = {
        {{18082, 57692, 41793, 56495},
         {2260, 7211, 5224, 7061},
         {282, 901, 653, 882},
         {17, 56, 40, 55},
         {2, 7, 5, 6},
         {0, 0, 0, 0}},
        {{8780, 52988, 13539, 19184},
         {1097, 6623, 1692, 2398},
         {137, 827, 211, 299},
         {8, 51, 13, 18},
         {1, 6, 1, 2},
         {0, 0, 0, 0}},
        {{63422, 13365, 41288, 19151},
         {7927, 1670, 5161, 2393},
         {990, 208, 645, 299},
         {61, 13, 40, 18},
         {7, 1, 5, 2},
         {0, 0, 0, 0}},
        {{52253, 3308, 26061, 28915},
         {6531, 413, 3257, 3614},
         {816, 51, 407, 451},
         {51, 3, 25, 28},
         {6, 0, 3, 3},
         {0, 0, 0, 0}},
        {{20395, 60753, 242, 16329},
         {2549, 7594, 30, 2041},
         {318, 949, 3, 255},
         {19, 59, 0, 15},
         {2, 7, 0, 1},
         {0, 0, 0, 0}},
        {{50722, 1293, 65018, 47184},
         {6340, 161, 8127, 5898},
         {792, 20, 1015, 737},
         {49, 1, 63, 46},
         {6, 0, 7, 5},
         {0, 0, 0, 0}},
        {{34292, 15596, 47910, 17286},
         {4286, 1949, 5988, 2160},
         {535, 243, 748, 270},
         {33, 15, 46, 16},
         {4, 1, 5, 2},
         {0, 0, 0, 0}},
        {{29575, 21839, 17112, 33989},
         {3696, 2729, 2139, 4248},
         {462, 341, 267, 531},
         {28, 21, 16, 33},
         {3, 2, 2, 4},
         {0, 0, 0, 0}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint16x4_t a = vld1_u16(test_vec[i].a);

        uint16x4_t r3 = vshr_n_u16(a, 3);
        uint16x4_t r6 = vshr_n_u16(a, 6);
        uint16x4_t r10 = vshr_n_u16(a, 10);
        uint16x4_t r13 = vshr_n_u16(a, 13);
        /* uint16x4_t r16 = vshr_n_u16(a, 16); */

        uint16x4_t check3 = vld1_u16(test_vec[i].r3);
        uint16x4_t check6 = vld1_u16(test_vec[i].r6);
        uint16x4_t check10 = vld1_u16(test_vec[i].r10);
        uint16x4_t check13 = vld1_u16(test_vec[i].r13);
        /* uint16x4_t check16 = vld1_u16(test_vec[i].r16); */

        ASSERT_EQUAL(r3, check3);
        ASSERT_EQUAL(r6, check6);
        ASSERT_EQUAL(r10, check10);
        ASSERT_EQUAL(r13, check13);
        /* ASSERT_EQUAL(r16, check16); */
    }

    return 0;
}
