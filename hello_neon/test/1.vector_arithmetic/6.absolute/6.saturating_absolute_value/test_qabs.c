// 2023-04-17 16:48
#include <neon.h>
#include <neon_test.h>
#include <stdint.h>
// int8x8_t vqabs_s8(int8x8_t a)
// int16x4_t vqabs_s16(int16x4_t a)
// int32x2_t vqabs_s32(int32x2_t a)
// int64x1_t vqabs_s64(int64x1_t a)
//
// int8x16_t vqabsq_s8(int8x16_t a)
// int16x8_t vqabsq_s16(int16x8_t a)
// int32x4_t vqabsq_s32(int32x4_t a)
// int64x2_t vqabsq_s64(int64x2_t a)
// ---------------------------------
// int8_t vqabsb_s8(int8_t a)
//             ^---int8
// int16_t vqabsh_s16(int16_t a)
//              ^---HI, int16
// int32_t vqabss_s32(int32_t a)
// int64_t vqabsd_s64(int64_t a)

TEST_CASE(test_vqabsb_s8) {
    static const struct {
        int8_t a;
        int8_t r;
    } test_vec[] = {
        {INT8_MIN, INT8_MAX},
        {23, 23},
        {79, 79},
        {-44, 44},
        {56, 56},
        {3, 3},
        {-28, 28},
        {10, 10}};
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8_t r = vqabsb_s8(test_vec[i].a);
        int8_t check = test_vec[i].r;
        ASSERT_EQUAL_SCALAR(r, check);
    }
    return 0;
}
