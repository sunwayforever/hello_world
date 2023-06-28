// 2023-04-20 16:48
#include <neon.h>
#include <neon_test.h>
// int8x8_t vqneg_s8(int8x8_t a)
// int16x4_t vqneg_s16(int16x4_t a)
// int32x2_t vqneg_s32(int32x2_t a)
// int64x1_t vqneg_s64(int64x1_t a)
//
// int8x16_t vqnegq_s8(int8x16_t a)
// int16x8_t vqnegq_s16(int16x8_t a)
// int32x4_t vqnegq_s32(int32x4_t a)
// int64x2_t vqnegq_s64(int64x2_t a)
// ---------------------------------
// int8_t vqnegb_s8(int8_t a)
// int16_t vqnegh_s16(int16_t a)
// int32_t vqnegs_s32(int32_t a)
// int64_t vqnegd_s64(int64_t a)

TEST_CASE(test_vqnegb_s8) {
    static const struct {
        int8_t a;
        int8_t r;
    } test_vec[] = {
        {INT8_MIN, INT8_MAX},
        {-59, 59},
        {-53, 53},
        {96, -96},
        {75, -75},
        {-55, 55},
        {-47, 47},
        {-61, 61}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8_t r = vqnegb_s8(test_vec[i].a);
        ASSERT_EQUAL_SCALAR(r, test_vec[i].r);
    }
    return 0;
}

TEST_CASE(test_vqneg) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{INT8_MIN, 92, -48, 20, 55, 1, 44, 90},
         {INT8_MAX, -92, 48, -20, -55, -1, -44, -90}},
        {{-56, 96, -3, 125, INT8_MIN, 80, -75, -2},
         {56, -96, 3, -125, INT8_MAX, -80, 75, 2}},
        {{2, -40, -121, INT8_MIN, -72, 115, -65, 108},
         {-2, 40, 121, INT8_MAX, 72, -115, 65, -108}},
        {{112, -88, -96, 16, INT8_MIN, -3, -32, -95},
         {-112, 88, 96, -16, INT8_MAX, 3, 32, 95}},
        {{-31, -51, -114, 25, INT8_MIN, -18, 23, 19},
         {31, 51, 114, -25, INT8_MAX, 18, -23, -19}},
        {{103, -56, INT8_MIN, -109, -54, 98, 27, -47},
         {-103, 56, INT8_MAX, 109, 54, -98, -27, 47}},
        {{-114, -112, -121, -30, INT8_MIN, 47, -126, 16},
         {114, 112, 121, 30, INT8_MAX, -47, 126, -16}},
        {{INT8_MIN, -16, 93, -77, -47, 42, 65, -21},
         {INT8_MAX, 16, -93, 77, 47, -42, -65, 21}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vqneg_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
