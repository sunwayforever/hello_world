// 2023-04-20 10:26
#include <neon.h>
#include <neon_test.h>
//                 v----narrow
// uint8x8_t vqshrun_n_s16(int16x8_t a,const int n)
//                ^---unsigned
// uint16x4_t vqshrun_n_s32(int32x4_t a,const int n)
// uint32x2_t vqshrun_n_s64(int64x2_t a,const int n)
//
// int8x8_t vqshrn_n_s16(int16x8_t a,const int n)
// int16x4_t vqshrn_n_s32(int32x4_t a,const int n)
// int32x2_t vqshrn_n_s64(int64x2_t a,const int n)
// uint8x8_t vqshrn_n_u16(uint16x8_t a,const int n)
// uint16x4_t vqshrn_n_u32(uint32x4_t a,const int n)
// uint32x2_t vqshrn_n_u64(uint64x2_t a,const int n)
// --------------------------------------------------
// scalar:
// uint8_t vqshrunh_n_s16(int16_t a,const int n)
//              ^---unsigned
// uint16_t vqshruns_n_s32(int32_t a,const int n)
// uint32_t vqshrund_n_s64(int64_t a,const int n)
//
// int8_t vqshrnh_n_s16(int16_t a,const int n)
//              ^---scalar
// int16_t vqshrns_n_s32(int32_t a,const int n)
// int32_t vqshrnd_n_s64(int64_t a,const int n)
// uint8_t vqshrnh_n_u16(uint16_t a,const int n)
// uint16_t vqshrns_n_u32(uint32_t a,const int n)
// uint32_t vqshrnd_n_u64(uint64_t a,const int n)
// ----------------------------------------------------------------
// high:
// uint8x16_t vqshrun_high_n_s16(uint8x8_t r,int16x8_t a,const int n)
//                 ^---unsigned
// uint16x8_t vqshrun_high_n_s32(uint16x4_t r,int32x4_t a,const int n)
// uint32x4_t vqshrun_high_n_s64(uint32x2_t r,int64x2_t a,const int n)
//
// int8x16_t vqshrn_high_n_s16(int8x8_t r,int16x8_t a,const int n)
// int16x8_t vqshrn_high_n_s32(int16x4_t r,int32x4_t a,const int n)
// int32x4_t vqshrn_high_n_s64(int32x2_t r,int64x2_t a,const int n)
// uint8x16_t vqshrn_high_n_u16(uint8x8_t r,uint16x8_t a,const int n)
// uint16x8_t vqshrn_high_n_u32(uint16x4_t r,uint32x4_t a,const int n)
// uint32x4_t vqshrn_high_n_u64(uint32x2_t r,uint64x2_t a,const int n)

TEST_CASE(test_vqshrun_n_s16) {
    static const struct {
        int16_t a[8];
        uint8_t r1[8];
        uint8_t r3[8];
        uint8_t r5[8];
        uint8_t r6[8];
        uint8_t r8[8];
    } test_vec[] = {
        {{16778, -3511, 3623, -12553, -21552, 14428, -7152, -804},
         {UINT8_MAX, 0, UINT8_MAX, 0, 0, UINT8_MAX, 0, 0},
         {UINT8_MAX, 0, UINT8_MAX, 0, 0, UINT8_MAX, 0, 0},
         {UINT8_MAX, 0, 113, 0, 0, UINT8_MAX, 0, 0},
         {UINT8_MAX, 0, 56, 0, 0, 225, 0, 0},
         {65, 0, 14, 0, 0, 56, 0, 0}},
        {{-24341, 22691, 19770, -13268, -21048, -6044, 26120, -28048},
         {0, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX, 0},
         {0, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX, 0},
         {0, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX, 0},
         {0, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX, 0},
         {0, 88, 77, 0, 0, 0, 102, 0}},
        {{-18009, -12667, 31944, -26468, -2009, 14544, -21284, -14284},
         {0, 0, UINT8_MAX, 0, 0, UINT8_MAX, 0, 0},
         {0, 0, UINT8_MAX, 0, 0, UINT8_MAX, 0, 0},
         {0, 0, UINT8_MAX, 0, 0, UINT8_MAX, 0, 0},
         {0, 0, UINT8_MAX, 0, 0, 227, 0, 0},
         {0, 0, 124, 0, 0, 56, 0, 0}},
        {{-10419, -30944, 19748, -5036, -18182, 725, 17694, -14956},
         {0, 0, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, 0},
         {0, 0, UINT8_MAX, 0, 0, 90, UINT8_MAX, 0},
         {0, 0, UINT8_MAX, 0, 0, 22, UINT8_MAX, 0},
         {0, 0, UINT8_MAX, 0, 0, 11, UINT8_MAX, 0},
         {0, 0, 77, 0, 0, 2, 69, 0}},
        {{6655, -14445, 12438, -17057, 12328, 1525, 10716, 10701},
         {UINT8_MAX, 0, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX},
         {UINT8_MAX, 0, UINT8_MAX, 0, UINT8_MAX, 190, UINT8_MAX, UINT8_MAX},
         {207, 0, UINT8_MAX, 0, UINT8_MAX, 47, UINT8_MAX, UINT8_MAX},
         {103, 0, 194, 0, 192, 23, 167, 167},
         {25, 0, 48, 0, 48, 5, 41, 41}},
        {{-4864, 9393, 1338, 13329, -6467, -9418, -13525, 10912},
         {0, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX},
         {0, UINT8_MAX, 167, UINT8_MAX, 0, 0, 0, UINT8_MAX},
         {0, UINT8_MAX, 41, UINT8_MAX, 0, 0, 0, UINT8_MAX},
         {0, 146, 20, 208, 0, 0, 0, 170},
         {0, 36, 5, 52, 0, 0, 0, 42}},
        {{13284, 31473, 20835, -29640, 11649, 23953, 24151, 22407},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX},
         {207, UINT8_MAX, UINT8_MAX, 0, 182, UINT8_MAX, UINT8_MAX, UINT8_MAX},
         {51, 122, 81, 0, 45, 93, 94, 87}},
        {{14411, -31108, -29379, -1350, -3725, -24875, 30140, -24375},
         {UINT8_MAX, 0, 0, 0, 0, 0, UINT8_MAX, 0},
         {UINT8_MAX, 0, 0, 0, 0, 0, UINT8_MAX, 0},
         {UINT8_MAX, 0, 0, 0, 0, 0, UINT8_MAX, 0},
         {225, 0, 0, 0, 0, 0, UINT8_MAX, 0},
         {56, 0, 0, 0, 0, 0, 117, 0}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x8_t a = vld1q_s16(test_vec[i].a);

        uint8x8_t r1 = vqshrun_n_s16(a, 1);
        uint8x8_t r3 = vqshrun_n_s16(a, 3);
        uint8x8_t r5 = vqshrun_n_s16(a, 5);
        uint8x8_t r6 = vqshrun_n_s16(a, 6);
        /* uint8x8_t r8 = vqshrun_n_s16(a, 8); */

        uint8x8_t check1 = vld1_u8(test_vec[i].r1);
        uint8x8_t check3 = vld1_u8(test_vec[i].r3);
        uint8x8_t check5 = vld1_u8(test_vec[i].r5);
        uint8x8_t check6 = vld1_u8(test_vec[i].r6);
        /* uint8x8_t check8 = vld1_u8(test_vec[i].r8); */

        ASSERT_EQUAL(r1, check1);
        ASSERT_EQUAL(r3, check3);
        ASSERT_EQUAL(r5, check5);
        ASSERT_EQUAL(r6, check6);
        /* ASSERT_EQUAL(r8, check8); */
    }
    return 0;
}

TEST_CASE(test_vqshrn_n_s16) {
    static const struct {
        int16_t a[8];
        int8_t r1[8];
        int8_t r3[8];
        int8_t r5[8];
        int8_t r6[8];
        int8_t r8[8];
    } test_vec[] = {
        {{11473, 27542, 24074, 2993, -19599, 550, 9579, -30586},
         {INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, 68, INT8_MAX,
          INT8_MIN},
         {INT8_MAX, INT8_MAX, INT8_MAX, 93, INT8_MIN, 17, INT8_MAX, INT8_MIN},
         {INT8_MAX, INT8_MAX, INT8_MAX, 46, INT8_MIN, 8, INT8_MAX, INT8_MIN},
         {44, 107, 94, 11, -77, 2, 37, -120}},
        {{17084, -6524, 32126, 25595, -28938, -21879, 9062, 14325},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MAX},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MAX},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MAX},
         {INT8_MAX, -102, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MAX},
         {66, -26, 125, 99, -114, -86, 35, 55}},
        {{-29873, 22946, 21481, 23396, -29946, 29021, -6992, 27898},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, -110,
          INT8_MAX},
         {-117, 89, 83, 91, -117, 113, -28, 108}},
        {{32294, -23469, 20219, -3832, -28195, 17308, -28236, 890},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN,
          111},
         {INT8_MAX, INT8_MIN, INT8_MAX, -120, INT8_MIN, INT8_MAX, INT8_MIN, 27},
         {INT8_MAX, INT8_MIN, INT8_MAX, -60, INT8_MIN, INT8_MAX, INT8_MIN, 13},
         {126, -92, 78, -15, -111, 67, -111, 3}},
        {{7196, 1372, -16017, 30048, -16820, -794, -8030, -14232},
         {INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, -100, INT8_MIN,
          INT8_MIN},
         {INT8_MAX, 42, INT8_MIN, INT8_MAX, INT8_MIN, -25, INT8_MIN, INT8_MIN},
         {112, 21, INT8_MIN, INT8_MAX, INT8_MIN, -13, -126, INT8_MIN},
         {28, 5, -63, 117, -66, -4, -32, -56}},
        {{-17570, 22893, 29962, -6325, -6394, -17622, -23432, -27458},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, -99, -100, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {-69, 89, 117, -25, -25, -69, -92, -108}},
        {{7104, 12185, -1316, 10404, -30024, 23076, -29589, -14046},
         {INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MIN},
         {INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MIN},
         {INT8_MAX, INT8_MAX, -42, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MIN},
         {111, INT8_MAX, -21, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN},
         {27, 47, -6, 40, -118, 90, -116, -55}},
        {{-28856, 21027, 28164, 2873, 25429, -12858, -31737, -14495},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, 89, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MAX, INT8_MAX, 44, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {-113, 82, 110, 11, 99, -51, -124, -57}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x8_t a = vld1q_s16(test_vec[i].a);

        int8x8_t r1 = vqshrn_n_s16(a, 1);
        int8x8_t r3 = vqshrn_n_s16(a, 3);
        int8x8_t r5 = vqshrn_n_s16(a, 5);
        int8x8_t r6 = vqshrn_n_s16(a, 6);
        /* int8x8_t r8 = vqshrn_n_s16(a, 8); */

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
