// 2023-04-20 15:33
#include <neon.h>
#include <neon_test.h>
// int8x8_t vqmovn_s16(int16x8_t a)
// int16x4_t vqmovn_s32(int32x4_t a)
// int32x2_t vqmovn_s64(int64x2_t a)
// uint8x8_t vqmovn_u16(uint16x8_t a)
// uint16x4_t vqmovn_u32(uint32x4_t a)
// uint32x2_t vqmovn_u64(uint64x2_t a)
//
// int8x16_t vqmovn_high_s16(int8x8_t r,int16x8_t a)
// int16x8_t vqmovn_high_s32(int16x4_t r,int32x4_t a)
// int32x4_t vqmovn_high_s64(int32x2_t r,int64x2_t a)
// uint8x16_t vqmovn_high_u16(uint8x8_t r,uint16x8_t a)
// uint16x8_t vqmovn_high_u32(uint16x4_t r,uint32x4_t a)
// uint32x4_t vqmovn_high_u64(uint32x2_t r,uint64x2_t a)
// ------------------------------------
// scalar:
// int8_t vqmovnh_s16(int16_t a)
// int16_t vqmovns_s32(int32_t a)
// int32_t vqmovnd_s64(int64_t a)
// uint8_t vqmovnh_u16(uint16_t a)
// uint16_t vqmovns_u32(uint32_t a)
// uint32_t vqmovnd_u64(uint64_t a)
// ------------------------------------
// unsigned:
// uint8x8_t vqmovun_s16(int16x8_t a)
// uint16x4_t vqmovun_s32(int32x4_t a)
// uint32x2_t vqmovun_s64(int64x2_t a)
// uint8_t vqmovunh_s16(int16_t a)
// uint16_t vqmovuns_s32(int32_t a)
// uint32_t vqmovund_s64(int64_t a)
//
// uint8x16_t vqmovun_high_s16(uint8x8_t r,int16x8_t a)
// uint16x8_t vqmovun_high_s32(uint16x4_t r,int32x4_t a)
// uint32x4_t vqmovun_high_s64(uint32x2_t r,int64x2_t a)

TEST_CASE(test_vqmovn_s16) {
    static const struct {
        int16_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{12, 2618, 1578, -3171, 0, 4882, -13300, 1},
         {12, INT8_MAX, INT8_MAX, INT8_MIN, 0, INT8_MAX, INT8_MIN, 1}},
        {{599, -43, -27285, -97, -3, 3, 86, -2810},
         {INT8_MAX, -43, INT8_MIN, -97, -3, 3, 86, INT8_MIN}},
        {{-21, -1, 201, 58, 0, 2864, -10, -32766},
         {-21, -1, INT8_MAX, 58, 0, INT8_MAX, -10, INT8_MIN}},
        {{918, 44, -93, -1357, 623, 1, 4, 1},
         {INT8_MAX, 44, -93, INT8_MIN, INT8_MAX, 1, 4, 1}},
        {{106, 7840, 19948, -618, -23, -408, -1, 2676},
         {106, INT8_MAX, INT8_MAX, INT8_MIN, -23, INT8_MIN, -1, INT8_MAX}},
        {{10178, 29083, -1, 108, 179, -217, 1, 0},
         {INT8_MAX, INT8_MAX, -1, 108, INT8_MAX, INT8_MIN, 1, 0}},
        {{4038, -1, 0, 4, -602, -63, -4, -3598},
         {INT8_MAX, -1, 0, 4, INT8_MIN, -63, -4, INT8_MIN}},
        {{-1871, -51, 209, 23, 118, -4, 168, -40},
         {INT8_MIN, -51, INT8_MAX, 23, 118, -4, INT8_MAX, -40}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x8_t a = vld1q_s16(test_vec[i].a);
        int8x8_t r = vqmovn_s16(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vqmovun_s16) {
    static const struct {
        int16_t a[8];
        uint8_t r[8];
    } test_vec[] = {
        {{-18345, 7399, -5353, -25148, -27188, 13769, 990, 9688},
         {0, UINT8_MAX, 0, 0, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX}},
        {{27133, 15294, -22736, 17779, 32692, 10966, 17328, 1930},
         {UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX}},
        {{29179, 4643, -6308, 10671, 30844, 23134, 13948, 31103},
         {UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX,
          UINT8_MAX}},
        {{16031, -12364, 10213, -26091, -5210, 22468, 20014, 10590},
         {UINT8_MAX, 0, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX}},
        {{-32064, 7227, -5527, -6587, -23709, -8384, -16167, 30808},
         {0, UINT8_MAX, 0, 0, 0, 0, 0, UINT8_MAX}},
        {{3326, -7352, 23859, -9859, 16712, 30256, -28784, 20639},
         {UINT8_MAX, 0, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX}},
        {{-9711, 31340, -19772, 10080, -24235, 12038, 24161, 24487},
         {0, UINT8_MAX, 0, UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX}},
        {{-4246, -25278, -16308, -27529, -22783, -28406, -22218, 18401},
         {0, 0, 0, 0, 0, 0, 0, UINT8_MAX}},

    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x8_t a = vld1q_s16(test_vec[i].a);
        uint8x8_t r = vqmovun_s16(a);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
