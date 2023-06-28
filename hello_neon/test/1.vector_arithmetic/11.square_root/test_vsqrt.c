// 2023-04-18 11:31
#include <neon.h>
#include <neon_test.h>
// float32x2_t vsqrt_f32(float32x2_t a)
// float64x1_t vsqrt_f64(float64x1_t a)
//
// float32x4_t vsqrtq_f32(float32x4_t a)
// float64x2_t vsqrtq_f64(float64x2_t a)

TEST_CASE(test_vsqrt_f32) {
    static const struct {
        float a[2];
        float r[2];
    } test_vec[] = {
        {{418.49, 138.55}, {20.45, 11.77}}, {{853.74, 358.34}, {29.21, 18.92}},
        {{14.18, 322.90}, {3.76, 17.96}},   {{96.91, 205.64}, {9.84, 14.34}},
        {{916.53, 885.37}, {30.27, 29.75}}, {{819.13, 247.72}, {28.62, 15.73}},
        {{663.07, 181.34}, {25.75, 13.46}}, {{895.59, 39.87}, {29.92, 6.31}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t r = vsqrt_f32(a);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}
