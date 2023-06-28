// 2023-04-18 10:59
#include <neon.h>
#include <neon_test.h>
// uint32x2_t vrsqrte_u32(uint32x2_t a)
//             ^^^^^^---reciprocal sqrt estimate, r[i]=sqrt(1/a[i])
// float32x2_t vrsqrte_f32(float32x2_t a)
// float64x1_t vrsqrte_f64(float64x1_t a)
//
// uint32x4_t vrsqrteq_u32(uint32x4_t a)
// float32x4_t vrsqrteq_f32(float32x4_t a)
// float64x2_t vrsqrteq_f64(float64x2_t a)
// ---------------------------------------
// float32_t vrsqrtes_f32(float32_t a)
// float64_t vrsqrted_f64(float64_t a)
// ----------------------------------------------------
// float32x2_t vrsqrts_f32(float32x2_t a,float32x2_t b)
// float64x1_t vrsqrts_f64(float64x1_t a,float64x1_t b)
// float32x4_t vrsqrtsq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vrsqrtsq_f64(float64x2_t a,float64x2_t b)
// -----------------------------------------------------
// float32_t vrsqrtss_f32(float32_t a,float32_t b)
// float64_t vrsqrtsd_f64(float64_t a,float64_t b)

TEST_CASE(test_vrsqrtes_f32) {
    static const struct {
        float a;
        float r;
    } test_vec[] = {{33.60, 0.17}, {85.84, 0.11}, {3.60, 0.53},  {82.23, 0.11},
                    {13.21, 0.28}, {98.57, 0.10}, {20.89, 0.22}, {58.72, 0.13}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float a = test_vec[i].a;
        float r = vrsqrtes_f32(a);
        ASSERT_CLOSE_SCALAR(r, test_vec[i].r);
    }
    return 0;
}
