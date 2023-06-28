// 2023-04-18 14:45
#include <neon.h>
#include <neon_test.h>
// int8_t vaddv_s8(int8x8_t a)
//            ^---across vector, r=sum(a[i])
// int16_t vaddv_s16(int16x4_t a)
// int32_t vaddv_s32(int32x2_t a)
// uint8_t vaddv_u8(uint8x8_t a)
// uint16_t vaddv_u16(uint16x4_t a)
// uint32_t vaddv_u32(uint32x2_t a)
//
// int8_t vaddvq_s8(int8x16_t a)
// int16_t vaddvq_s16(int16x8_t a)
// int32_t vaddvq_s32(int32x4_t a)
// int64_t vaddvq_s64(int64x2_t a)
// uint8_t vaddvq_u8(uint8x16_t a)
// uint16_t vaddvq_u16(uint16x8_t a)
// uint32_t vaddvq_u32(uint32x4_t a)
// ----------------------------------
// float32_t vaddv_f32(float32x2_t a)
// float32_t vaddvq_f32(float32x4_t a)
// float64_t vaddvq_f64(float64x2_t a)

TEST_CASE(test_vaddv_s8) {
    static const struct {
        int8_t a[8];
        int8_t r;
    } test_vec[] = {
        {{-38, -113, -89, 100, 121, 62, 96, 114}, -3},
        {{54, 22, 118, 46, 72, 40, 123, -76}, -113},
        {{-45, 36, 32, -99, 35, 22, -20, 18}, -21},
        {{24, 119, 80, -27, 57, 66, 123, 20}, -50},
        {{-46, 35, 120, 75, 97, -40, -67, -104}, 70},
        {{-17, 51, -58, 55, 91, 65, -20, 46}, -43},
        {{102, 12, -52, -119, 34, -72, -100, 58}, 119},
        {{48, -20, 31, 105, 47, -101, 125, 1}, -20}};
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8_t r = vaddv_s8(a);
        ASSERT_EQUAL_SCALAR(r, test_vec[i].r);
    }
    return 0;
}
