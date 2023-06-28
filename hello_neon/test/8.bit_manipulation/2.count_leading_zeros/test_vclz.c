// 2023-04-20 17:58
#include <neon.h>
#include <neon_test.h>
// int8x8_t vclz_s8(int8x8_t a)
// int16x4_t vclz_s16(int16x4_t a)
// int32x2_t vclz_s32(int32x2_t a)
// uint8x8_t vclz_u8(uint8x8_t a)
// uint16x4_t vclz_u16(uint16x4_t a)
// uint32x2_t vclz_u32(uint32x2_t a)
//
// int8x16_t vclzq_s8(int8x16_t a)
// int16x8_t vclzq_s16(int16x8_t a)
// int32x4_t vclzq_s32(int32x4_t a)
// uint8x16_t vclzq_u8(uint8x16_t a)
// uint16x8_t vclzq_u16(uint16x8_t a)
// uint32x4_t vclzq_u32(uint32x4_t a)

TEST_CASE(test_vclz_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{0, -1, 26, -5, -62, 9, -9, 26}, {8, 0, 3, 0, 0, 4, 0, 3}},
        {{-87, 0, 2, -4, -15, -3, -2, 4}, {0, 8, 6, 0, 0, 0, 0, 5}},
        {{49, -8, -26, -7, -2, -13, 5, 1}, {2, 0, 0, 0, 0, 0, 5, 7}},
        {{-30, -1, -25, 30, -5, 84, -4, -1}, {0, 0, 0, 3, 0, 1, 0, 0}},
        {{36, -117, -3, 1, 5, 3, 44, 15}, {2, 0, 0, 7, 5, 6, 2, 4}},
        {{0, 0, -115, -2, -2, 1, -18, 0}, {8, 8, 0, 0, 0, 7, 0, 8}},
        {{-12, -6, -2, -1, -2, -1, 58, 0}, {0, 0, 0, 0, 0, 0, 2, 8}},
        {{-2, 0, 117, 0, 2, 12, 1, 0}, {0, 8, 1, 8, 6, 4, 7, 8}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vclz_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
