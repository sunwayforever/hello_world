// 2023-04-17 17:11
#include <neon.h>
#include <neon_test.h>
/* NOTE: 和 add 基本相同, 但不支持 {s,u}64 */
// int8x8_t vmax_s8(int8x8_t a,int8x8_t b)
// int16x4_t vmax_s16(int16x4_t a,int16x4_t b)
// int32x2_t vmax_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vmax_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vmax_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vmax_u32(uint32x2_t a,uint32x2_t b)
//
// int8x16_t vmaxq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vmaxq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vmaxq_s32(int32x4_t a,int32x4_t b)
// uint8x16_t vmaxq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vmaxq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vmaxq_u32(uint32x4_t a,uint32x4_t b)
// -------------------------------------------------
// float32x2_t vmax_f32(float32x2_t a,float32x2_t b)
// float64x1_t vmax_f64(float64x1_t a,float64x1_t b)
//
// float32x4_t vmaxq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vmaxq_f64(float64x2_t a,float64x2_t b)
#include <math.h>

/* 针对 nanf, vmax 与 libm 中的 fmax 行为不一致, vmaxnm 与 fmax 是一致的 */
TEST_CASE(test_vmax_f32) {
    float nan = nanf("");
    static struct {
        float a[2];
        float b[2];
        float r[2];
    } test_vec[8] = {
        {{0, -464.60}, {866.05, 0}, {0, 0}},
        {{0, 861.67}, {0, 861.67}, {0, 861.67}},
        {{378.04, -897.72}, {-584.86, 922.34}, {378.04, 922.34}},
        {{169.18, 164.66}, {295.66, -857.49}, {295.66, 164.66}},
        {{-597.22, -740.42}, {-439.12, -673.24}, {-439.12, -673.24}},
        {{693.53, -114.27}, {599.27, 359.67}, {693.53, 359.67}},
        {{-598.01, -64.73}, {384.43, 446.35}, {384.43, 446.35}},
        {{449.76, 326.28}, {146.92, -725.29}, {449.76, 326.28}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        for (int j = 0; j < 2; j++) {
            if (test_vec[i].a[j] == 0) {
                test_vec[i].a[j] = nan;
            }
            if (test_vec[i].b[j] == 0) {
                test_vec[i].b[j] = nan;
            }
            if (test_vec[i].r[j] == 0) {
                test_vec[i].r[j] = nan;
            }
        }
    }

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        float32x2_t r = vmax_f32(a, b);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }

    return 0;
}
