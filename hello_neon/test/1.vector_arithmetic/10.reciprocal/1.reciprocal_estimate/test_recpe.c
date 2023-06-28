// 2023-04-18 10:37
#include <neon.h>
#include <neon_test.h>
// uint32x2_t vrecpe_u32(uint32x2_t a)
//             ^^^^^---reciprocal estimate, r[i]=1.0/a[i]
// float32x2_t vrecpe_f32(float32x2_t a)
// float64x1_t vrecpe_f64(float64x1_t a)
//
// float32x4_t vrecpeq_f32(float32x4_t a)
// uint32x4_t vrecpeq_u32(uint32x4_t a)
// float64x2_t vrecpeq_f64(float64x2_t a)
// -----------------------------------------
// float32_t vrecpes_f32(float32_t a)
//                 ^---scalar
// float64_t vrecped_f64(float64_t a)
// ---------------------------------------------------
// float32x2_t vrecps_f32(float32x2_t a,float32x2_t b)
//              ^^^^^---reciprocal step, r[i]=2.0-a[i]*b[i]
// float64x1_t vrecps_f64(float64x1_t a,float64x1_t b)
//
// float32x4_t vrecpsq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vrecpsq_f64(float64x2_t a,float64x2_t b)
// ----------------------------------------------------
// float32_t vrecpss_f32(float32_t a,float32_t b)
// float64_t vrecpsd_f64(float64_t a,float64_t b)

TEST_CASE(test_vrecpe_f32) {
    static const struct {
        float a[2];
        float r[2];
    } test_vec[] = {
        {{-3.61, -8.68}, {-0.28, -0.12}}, {{-6.51, -7.63}, {-0.15, -0.13}},
        {{-2.80, -7.27}, {-0.36, -0.14}}, {{-6.49, -7.56}, {-0.15, -0.13}},
        {{-5.41, -0.72}, {-0.18, -1.39}}, {{8.89, 2.37}, {0.11, 0.42}},
        {{-6.54, 4.78}, {-0.15, 0.21}},   {{7.48, 5.64}, {0.13, 0.18}},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t r = vrecpe_f32(a);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}

TEST_CASE(test_vrecps_f32) {
    static const struct {
        float a[2];
        float b[2];
        float r[2];
    } test_vec[] = {
        {{47.87, 269.61}, {96.45, -250.23}, {-4615.06, 67466.51}},
        {{-555.90, -609.45}, {322.74, -176.70}, {179413.17, -107687.81}},
        {{-980.02, 183.49}, {-974.10, 331.54}, {-954635.50, -60832.28}},
        {{388.02, -288.08}, {-277.25, 632.81}, {107580.54, 182301.89}},
        {{-326.10, 760.56}, {-335.74, -532.85}, {-109482.81, 405266.38}},
        {{-255.56, 505.21}, {-48.40, 933.85}, {-12367.10, -471788.34}},
        {{785.46, 215.93}, {289.25, 644.14}, {-227192.31, -139087.16}},
        {{763.47, 586.70}, {40.67, -188.67}, {-31048.32, 110694.69}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        float32x2_t r = vrecps_f32(a, b);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}
