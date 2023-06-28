// 2023-04-18 15:07
#include <neon.h>
#include <neon_test.h>
// int8_t vmaxv_s8(int8x8_t a)
// int16_t vmaxv_s16(int16x4_t a)
// int32_t vmaxv_s32(int32x2_t a)
// uint8_t vmaxv_u8(uint8x8_t a)
// uint16_t vmaxv_u16(uint16x4_t a)
// uint32_t vmaxv_u32(uint32x2_t a)
//
// int8_t vmaxvq_s8(int8x16_t a)
// int16_t vmaxvq_s16(int16x8_t a)
// int32_t vmaxvq_s32(int32x4_t a)
// uint8_t vmaxvq_u8(uint8x16_t a)
// uint16_t vmaxvq_u16(uint16x8_t a)
// uint32_t vmaxvq_u32(uint32x4_t a)
// -----------------------------------
// float32_t vmaxv_f32(float32x2_t a)
//
// float32_t vmaxvq_f32(float32x4_t a)
// float64_t vmaxvq_f64(float64x2_t a)
TEST_CASE(test_vmaxv_f32) {
    static const struct {
        float a[2];
        float r;
    } test_vec[] = {{{498.24, 700.18}, 700.18},  {{-550.14, -372.82}, -372.82},
                    {{-184.85, 347.23}, 347.23}, {{-183.13, 910.25}, 910.25},
                    {{995.08, 458.35}, 995.08},  {{954.33, 629.96}, 954.33},
                    {{-93.64, 684.43}, 684.43},  {{-76.95, -360.35}, -76.95}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float r = vmaxv_f32(a);
        ASSERT_EQUAL_SCALAR(r, test_vec[i].r);
    }
    return 0;
}
