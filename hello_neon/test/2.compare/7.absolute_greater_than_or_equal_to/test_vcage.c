// 2023-04-18 17:38
#include <neon.h>
#include <neon_test.h>
// uint32x2_t vcage_f32(float32x2_t a,float32x2_t b)
// uint64x1_t vcage_f64(float64x1_t a,float64x1_t b)
//
// uint32x4_t vcageq_f32(float32x4_t a,float32x4_t b)
// uint64x2_t vcageq_f64(float64x2_t a,float64x2_t b)
// --------------------------------------------------
// uint32_t vcages_f32(float32_t a,float32_t b)
// uint64_t vcaged_f64(float64_t a,float64_t b)

TEST_CASE(test_vcage_f32) {
    struct {
        float a[2];
        float b[2];
        uint32_t r[2];
    } test_vec[] = {
        {{311.69, -932.68}, {98.33, -552.98}, {UINT32_MAX, UINT32_MAX}},
        {{959.61, 617.75}, {-197.11, 562.98}, {UINT32_MAX, UINT32_MAX}},
        {{468.98, -916.49}, {965.35, 700.25}, {0, UINT32_MAX}},
        {{-647.13, -147.35}, {-117.68, -241.37}, {UINT32_MAX, 0}},
        {{-664.10, -976.12}, {874.22, -12.94}, {0, UINT32_MAX}},
        {{25.04, -125.75}, {212.15, 782.89}, {0, 0}},
        {{561.17, 217.87}, {-238.74, 679.32}, {UINT32_MAX, 0}},
        {{-965.46, -738.96}, {-711.74, 346.23}, {UINT32_MAX, UINT32_MAX}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        uint32x2_t r = vcage_f32(a, b);
        uint32x2_t check = vld1_u32(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
