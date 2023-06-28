// 2023-04-20 17:42
#include <neon.h>
#include <neon_test.h>
// int8x8_t vcls_s8(int8x8_t a)
// int16x4_t vcls_s16(int16x4_t a)
// int32x2_t vcls_s32(int32x2_t a)
// int8x8_t vcls_u8(uint8x8_t a)
// int16x4_t vcls_u16(uint16x4_t a)
// int32x2_t vcls_u32(uint32x2_t a)
//
// int8x16_t vclsq_s8(int8x16_t a)
// int16x8_t vclsq_s16(int16x8_t a)
// int32x4_t vclsq_s32(int32x4_t a)
// int8x16_t vclsq_u8(uint8x16_t a)
// int16x8_t vclsq_u16(uint16x8_t a)
// int32x4_t vclsq_u32(uint32x4_t a)

TEST_CASE(test_vcls_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{0, -1, -2, 6, -1, -4, 8, -1}, {7, 7, 6, 4, 7, 5, 3, 7}},
        {{-14, -4, -1, -1, -3, -97, 0, 14}, {3, 5, 7, 7, 5, 0, 7, 3}},
        {{1, -1, -1, -3, 12, -37, -48, -59}, {6, 7, 7, 5, 3, 1, 1, 1}},
        {{17, 24, -7, -2, 0, -13, 0, -1}, {2, 2, 4, 6, 7, 3, 7, 7}},
        {{-20, 0, 15, 0, -37, 4, -1, -16}, {2, 7, 3, 7, 1, 4, 7, 3}},
        {{-2, 0, 0, -102, -1, 1, 0, -19}, {6, 7, 7, 0, 7, 6, 7, 2}},
        {{-4, -27, -1, -7, 0, 0, 3, -3}, {5, 2, 7, 4, 7, 7, 5, 5}},
        {{2, -2, -12, -1, 1, 3, -3, 1}, {5, 6, 3, 7, 6, 5, 5, 6}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vcls_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vcls_u8) {
    static const struct {
        uint8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{0, UINT8_MAX, 113, 36, 14, 0, 3, 31}, {7, 7, 0, 1, 3, 7, 5, 2}},
        {{60, 30, 10, 63, 17, 22, 9, 4}, {1, 2, 3, 1, 2, 2, 3, 4}},
        {{17, 0, 4, 5, 10, 162, 180, 34}, {2, 7, 4, 4, 3, 0, 0, 1}},
        {{0, 2, 41, 21, 1, 6, 57, 0}, {7, 5, 1, 2, 6, 4, 1, 7}},
        {{39, 47, 1, 41, 0, 1, 14, 77}, {1, 1, 6, 1, 7, 6, 3, 0}},
        {{127, 47, 12, 93, 1, 15, 3, 26}, {0, 1, 3, 0, 6, 3, 5, 2}},
        {{1, 27, 4, 162, 29, 3, 15, 84}, {6, 2, 4, 0, 2, 5, 3, 0}},
        {{0, 41, 59, 3, 2, 181, 45, 121}, {7, 1, 1, 5, 5, 0, 1, 0}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint8x8_t a = vld1_u8(test_vec[i].a);
        int8x8_t r = vcls_u8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
