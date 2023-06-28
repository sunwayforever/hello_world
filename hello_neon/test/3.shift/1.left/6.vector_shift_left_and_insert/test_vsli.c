// 2023-04-19 15:42
#include <neon.h>
#include <neon_test.h>
// int8x8_t vsli_n_s8(int8x8_t a,int8x8_t b,const int n)
//           ^^^---shift left insert, r[i] = a & ((1 << n) - 1)+b << n;
// int16x4_t vsli_n_s16(int16x4_t a,int16x4_t b,const int n)
// int32x2_t vsli_n_s32(int32x2_t a,int32x2_t b,const int n)
// int64x1_t vsli_n_s64(int64x1_t a,int64x1_t b,const int n)
// uint8x8_t vsli_n_u8(uint8x8_t a,uint8x8_t b,const int n)
// uint16x4_t vsli_n_u16(uint16x4_t a,uint16x4_t b,const int n)
// uint32x2_t vsli_n_u32(uint32x2_t a,uint32x2_t b,const int n)
// uint64x1_t vsli_n_u64(uint64x1_t a,uint64x1_t b,const int n)
//
// int8x16_t vsliq_n_s8(int8x16_t a,int8x16_t b,const int n)
// int16x8_t vsliq_n_s16(int16x8_t a,int16x8_t b,const int n)
// int32x4_t vsliq_n_s32(int32x4_t a,int32x4_t b,const int n)
// int64x2_t vsliq_n_s64(int64x2_t a,int64x2_t b,const int n)
// uint8x16_t vsliq_n_u8(uint8x16_t a,uint8x16_t b,const int n)
// uint16x8_t vsliq_n_u16(uint16x8_t a,uint16x8_t b,const int n)
// uint32x4_t vsliq_n_u32(uint32x4_t a,uint32x4_t b,const int n)
// uint64x2_t vsliq_n_u64(uint64x2_t a,uint64x2_t b,const int n)
// --------------------------------------------------------------
// poly64x1_t vsli_n_p64(poly64x1_t a,poly64x1_t b,const int n)
// poly8x8_t vsli_n_p8(poly8x8_t a,poly8x8_t b,const int n)
// poly16x4_t vsli_n_p16(poly16x4_t a,poly16x4_t b,const int n)
//
// poly64x2_t vsliq_n_p64(poly64x2_t a,poly64x2_t b,const int n)
// poly8x16_t vsliq_n_p8(poly8x16_t a,poly8x16_t b,const int n)
// poly16x8_t vsliq_n_p16(poly16x8_t a,poly16x8_t b,const int n)
// --------------------------------------------------------------
// int64_t vslid_n_s64(int64_t a,int64_t b,const int n)
// uint64_t vslid_n_u64(uint64_t a,uint64_t b,const int n)

TEST_CASE(test_vsli_s16) {
    struct {
        int16_t a[4];
        int16_t b[4];
        int16_t r[4];
    } test_vec[] = {
        {{22332, -2389, -6176, 24298},
         {-11, -12, 535, 14},
         {-84, -93, 4280, 114}},
        {{-30833, -3392, 7263, 1769},
         {15, -10006, -26926, -19506},
         {127, -14512, -18793, -24975}},
        {{24162, 31020, -13216, -28236},
         {9, 11, 16, 2895},
         {74, 92, 128, 23164}},
        {{-844, -30766, -24430, 3898},
         {9, -9, -16, 2356},
         {76, -70, -126, 18850}},
        {{7438, -17736, 2143, -16698},
         {-10, -11, -15, 12},
         {-74, -88, -113, 102}},
        {{2818, -21698, 29388, 22965},
         {10, 27061, 13, -13277},
         {82, 19886, 108, 24861}},
        {{17345, -26964, 10446, -1539},
         {14, -7811, -9, 19084},
         {113, 3052, -66, 21605}},
        {{22929, -270, 5830, 15563},
         {10, -15886, -13, 12},
         {81, 3986, -98, 99}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int16x4_t r = vsli_n_s16(a, b, 3);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
