// 2023-04-17 17:17
#include <neon.h>
#include <neon_test.h>
// int8x8_t vmin_s8(int8x8_t a,int8x8_t b)
// int16x4_t vmin_s16(int16x4_t a,int16x4_t b)
// int32x2_t vmin_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vmin_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vmin_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vmin_u32(uint32x2_t a,uint32x2_t b)
//
// int8x16_t vminq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vminq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vminq_s32(int32x4_t a,int32x4_t b)
// uint8x16_t vminq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vminq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vminq_u32(uint32x4_t a,uint32x4_t b)
// --------------------------------------------------
// float32x2_t vmin_f32(float32x2_t a,float32x2_t b)
// float64x1_t vmin_f64(float64x1_t a,float64x1_t b)
// float32x4_t vminq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vminq_f64(float64x2_t a,float64x2_t b)
// ---------------------------------------------------
// float32x2_t vmaxnm_f32(float32x2_t a,float32x2_t b)
// float64x1_t vmaxnm_f64(float64x1_t a,float64x1_t b)
// float32x2_t vminnm_f32(float32x2_t a,float32x2_t b)
// float64x1_t vminnm_f64(float64x1_t a,float64x1_t b)
//
// float32x4_t vmaxnmq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vmaxnmq_f64(float64x2_t a,float64x2_t b)
// float32x4_t vminnmq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vminnmq_f64(float64x2_t a,float64x2_t b)

TEST_CASE(test_vmaxnm_f32) {
    float nan = nanf("");
    static struct {
        float a[2];
        float b[2];
        float r[2];
    } test_vec[] = {
        {{0, 656.90}, {427.79, 0}, {427.79, 656.90}},
        {{0, 116.96}, {0, -999.94}, {0, 116.96}},
        {{-619.20, -413.47}, {871.28, -660.33}, {871.28, -413.47}},
        {{422.55, 160.51}, {148.88, 905.13}, {422.55, 905.13}},
        {{-605.53, -971.47}, {182.75, -737.07}, {182.75, -737.07}},
        {{-182.06, -678.54}, {165.68, 413.12}, {165.68, 413.12}},
        {{20.28, -770.49}, {647.00, -632.40}, {647.00, -632.40}},
        {{949.17, 616.00}, {-967.88, -301.85}, {949.17, 616.00}}};

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
        float32x2_t r = vmaxnm_f32(a, b);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}
