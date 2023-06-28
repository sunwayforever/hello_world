// 2023-04-21 15:31
#include <neon.h>
#include <neon_test.h>
// int8x8_t vrev64_s8(int8x8_t vec)
// int16x4_t vrev64_s16(int16x4_t vec)
// int32x2_t vrev64_s32(int32x2_t vec)
// uint8x8_t vrev64_u8(uint8x8_t vec)
// uint16x4_t vrev64_u16(uint16x4_t vec)
// uint32x2_t vrev64_u32(uint32x2_t vec)
// float32x2_t vrev64_f32(float32x2_t vec)
// poly8x8_t vrev64_p8(poly8x8_t vec)
// poly16x4_t vrev64_p16(poly16x4_t vec)
// int8x8_t vrev32_s8(int8x8_t vec)
// int16x4_t vrev32_s16(int16x4_t vec)
// uint8x8_t vrev32_u8(uint8x8_t vec)
// uint16x4_t vrev32_u16(uint16x4_t vec)
// poly8x8_t vrev32_p8(poly8x8_t vec)
// poly16x4_t vrev32_p16(poly16x4_t vec)
// int8x8_t vrev16_s8(int8x8_t vec)
// uint8x8_t vrev16_u8(uint8x8_t vec)
// poly8x8_t vrev16_p8(poly8x8_t vec)
//
// int8x16_t vrev64q_s8(int8x16_t vec)
// int16x8_t vrev64q_s16(int16x8_t vec)
// int32x4_t vrev64q_s32(int32x4_t vec)
// uint8x16_t vrev64q_u8(uint8x16_t vec)
// uint16x8_t vrev64q_u16(uint16x8_t vec)
// uint32x4_t vrev64q_u32(uint32x4_t vec)
// float32x4_t vrev64q_f32(float32x4_t vec)
// poly8x16_t vrev64q_p8(poly8x16_t vec)
// poly16x8_t vrev64q_p16(poly16x8_t vec)
// int8x16_t vrev32q_s8(int8x16_t vec)
// int16x8_t vrev32q_s16(int16x8_t vec)
// uint8x16_t vrev32q_u8(uint8x16_t vec)
// uint16x8_t vrev32q_u16(uint16x8_t vec)
// poly8x16_t vrev32q_p8(poly8x16_t vec)
// poly16x8_t vrev32q_p16(poly16x8_t vec)
// int8x16_t vrev16q_s8(int8x16_t vec)
// uint8x16_t vrev16q_u8(uint8x16_t vec)
// poly8x16_t vrev16q_p8(poly8x16_t vec)

TEST_CASE(test_vrev16_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{-101, 94, -1, 100, -108, 34, 116, -90},
         {94, -101, 100, -1, 34, -108, -90, 116}},
        {{-10, 59, -1, 119, -58, 118, 36, -89},
         {59, -10, 119, -1, 118, -58, -89, 36}},
        {{92, -49, 6, -8, -93, -42, -88, -7},
         {-49, 92, -8, 6, -42, -93, -7, -88}},
        {{19, 0, 98, -77, -121, 123, 71, 34},
         {0, 19, -77, 98, 123, -121, 34, 71}},
        {{-39, 70, -122, 109, 104, -6, 20, 94},
         {70, -39, 109, -122, -6, 104, 94, 20}},
        {{54, 19, -43, -4, -119, -7, -92, -26},
         {19, 54, -4, -43, -7, -119, -26, -92}},
        {{-56, -86, -34, 107, -127, -121, 100, -108},
         {-86, -56, 107, -34, -121, -127, -108, 100}},
        {{-121, -58, 72, 14, 65, -113, 48, 26},
         {-58, -121, 14, 72, -113, 65, 26, 48}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vrev16_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
