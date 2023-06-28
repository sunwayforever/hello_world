// 2023-04-20 10:59
#include <neon.h>
#include <neon_test.h>

// uint8x8_t vqrshrun_n_s16(int16x8_t a,const int n)
//            ^^--s^^aturating
//             +---++rounding
//                 ++--unsigned
//                  +---narrow
// uint16x4_t vqrshrun_n_s32(int32x4_t a,const int n)
// uint32x2_t vqrshrun_n_s64(int64x2_t a,const int n)
// uint8_t vqrshrunh_n_s16(int16_t a,const int n)
// uint16_t vqrshruns_n_s32(int32_t a,const int n)
// uint32_t vqrshrund_n_s64(int64_t a,const int n)
//
// int8x8_t vqrshrn_n_s16(int16x8_t a,const int n)
// int16x4_t vqrshrn_n_s32(int32x4_t a,const int n)
// int32x2_t vqrshrn_n_s64(int64x2_t a,const int n)
// uint8x8_t vqrshrn_n_u16(uint16x8_t a,const int n)
// uint16x4_t vqrshrn_n_u32(uint32x4_t a,const int n)
// uint32x2_t vqrshrn_n_u64(uint64x2_t a,const int n)
// --------------------------------------------------
// scalar:
// int8_t vqrshrnh_n_s16(int16_t a,const int n)
// int16_t vqrshrns_n_s32(int32_t a,const int n)
// int32_t vqrshrnd_n_s64(int64_t a,const int n)
// uint8_t vqrshrnh_n_u16(uint16_t a,const int n)
// uint16_t vqrshrns_n_u32(uint32_t a,const int n)
// uint32_t vqrshrnd_n_u64(uint64_t a,const int n)
// ------------------------------------------------------------------
// uint8x16_t vqrshrun_high_n_s16(uint8x8_t r,int16x8_t a,const int n)
// uint16x8_t vqrshrun_high_n_s32(uint16x4_t r,int32x4_t a,const int n)
// uint32x4_t vqrshrun_high_n_s64(uint32x2_t r,int64x2_t a,const int n)
//
// int8x16_t vqrshrn_high_n_s16(int8x8_t r,int16x8_t a,const int n)
// int16x8_t vqrshrn_high_n_s32(int16x4_t r,int32x4_t a,const int n)
// int32x4_t vqrshrn_high_n_s64(int32x2_t r,int64x2_t a,const int n)
// uint8x16_t vqrshrn_high_n_u16(uint8x8_t r,uint16x8_t a,const int n)
// uint16x8_t vqrshrn_high_n_u32(uint16x4_t r,uint32x4_t a,const int n)
// uint32x4_t vqrshrn_high_n_u64(uint32x2_t r,uint64x2_t a,const int n)

TEST_CASE(test_vqrshrn_n_s16) {
    static const struct {
        int16_t a[8];
        int8_t r1[8];
        int8_t r3[8];
        int8_t r5[8];
        int8_t r6[8];
        int8_t r8[8];
    } test_vec[] = {
        {{-4618, -28204, 30834, 6009, 28312, -2262, -24784, 13790},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX, -71, INT8_MIN,
          INT8_MAX},
         {-72, INT8_MIN, INT8_MAX, 94, INT8_MAX, -35, INT8_MIN, INT8_MAX},
         {-18, -110, 120, 23, 111, -9, -97, 54}},
        {{5394, -25767, 6140, -10336, -32548, 8710, -30648, 16110},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {84, INT8_MIN, 96, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX},
         {21, -101, 24, -40, -127, 34, -120, 63}},
        {{-15754, -5937, 18491, -11265, 10678, -6198, -22327, -9444},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN,
          INT8_MIN},
         {INT8_MIN, -93, INT8_MAX, INT8_MIN, INT8_MAX, -97, INT8_MIN, INT8_MIN},
         {-62, -23, 72, -44, 42, -24, -87, -37}},
        {{30141, -18058, 6028, 26768, -26985, -8053, 31007, -27362},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MIN},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MIN},
         {INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX,
          INT8_MIN},
         {INT8_MAX, INT8_MIN, 94, INT8_MAX, INT8_MIN, -126, INT8_MAX, INT8_MIN},
         {118, -71, 24, 105, -105, -31, 121, -107}},
        {{-4549, 30333, 32054, -4791, 5030, 28628, -3908, 31051},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN,
          INT8_MAX},
         {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, -122,
          INT8_MAX},
         {-71, INT8_MAX, INT8_MAX, -75, 79, INT8_MAX, -61, INT8_MAX},
         {-18, 118, 125, -19, 20, 112, -15, 121}},
        {{-16027, -3789, -15400, 28761, -7078, 31056, 28253, -26354},
         {INT8_MIN, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, -118, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, -59, INT8_MIN, INT8_MAX, -111, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {-63, -15, -60, 112, -28, 121, 110, -103}},
        {{-29860, -27889, 22792, -20608, 21612, 10270, 26948, -22110},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {-117, -109, 89, -80, 84, 40, 105, -86}},
        {{-10965, 922, -3176, -3469, -15400, 13675, 31026, -28978},
         {INT8_MIN, INT8_MAX, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, 115, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MAX, INT8_MAX,
          INT8_MIN},
         {INT8_MIN, 29, -99, -108, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN},
         {INT8_MIN, 14, -50, -54, INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN},
         {-43, 4, -12, -14, -60, 53, 121, -113}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x8_t a = vld1q_s16(test_vec[i].a);

        int8x8_t r1 = vqrshrn_n_s16(a, 1);
        int8x8_t r3 = vqrshrn_n_s16(a, 3);
        int8x8_t r5 = vqrshrn_n_s16(a, 5);
        int8x8_t r6 = vqrshrn_n_s16(a, 6);
        /* int8x8_t r8 = vqrshrn_n_s16(a, 8); */

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
