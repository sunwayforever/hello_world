// 2023-04-17 18:22
#include <neon.h>
#include <neon_test.h>
//                      	          11.5	12.5	−11.5	−12.5
// to nearest, ties to even           12    12      −12.0   −12.0
// to nearest, ties away from zero    12    13      −12.0   −13.0
// toward 0                           11    12      −11.0   −12.0
// toward +∞                         12    13      −11.0   −12.0
// toward −∞                         11    12      −12.0   −13.0

// float32x2_t vrnd_f32(float32x2_t a)
//              ^^^---默认使用 Round to Integral, toward Zero (vector), 相当于
//              (int)x
// float64x1_t vrnd_f64(float64x1_t a)
// float32x4_t vrndq_f32(float32x4_t a)
// float64x2_t vrndq_f64(float64x2_t a)
//
// float32x2_t vrndn_f32(float32x2_t a)
//                 ^---Round to Integral, to nearest with ties to even
//                 (1.2->2.0, 2.2->2.0)
// float64x1_t vrndn_f64(float64x1_t a)
// float32x4_t vrndnq_f32(float32x4_t a)
// float64x2_t vrndnq_f64(float64x2_t a)
//
// float32_t vrndns_f32(float32_t a)
//
// float32x2_t vrndm_f32(float32x2_t a)
//                 ^---Round to Integral, toward Minus infinity (vector) (相当于
//                 floor)
// float64x1_t vrndm_f64(float64x1_t a)
// float32x4_t vrndmq_f32(float32x4_t a)
// float64x2_t vrndmq_f64(float64x2_t a)
//
// float32x2_t vrndp_f32(float32x2_t a)
//                 ^---Round to Integral, toward Plus infinity (vector) (相当于
//                 ceil)
// float64x1_t vrndp_f64(float64x1_t a)
// float32x4_t vrndpq_f32(float32x4_t a)
// float64x2_t vrndpq_f64(float64x2_t a)
//
// float32x2_t vrnda_f32(float32x2_t a)
//                 ^---Round to Integral, to nearest with ties to Away (vector)
//                 (四舍五入)
// float64x1_t vrnda_f64(float64x1_t a)
// float32x4_t vrndaq_f32(float32x4_t a)
// float64x2_t vrndaq_f64(float64x2_t a)
//
// float32x2_t vrndi_f32(float32x2_t a)
//                 ^---Round to Integral, using current rounding mode (vector).
// float64x1_t vrndi_f64(float64x1_t a)
// float32x4_t vrndiq_f32(float32x4_t a)
// float64x2_t vrndiq_f64(float64x2_t a)
//
// float32x2_t vrndx_f32(float32x2_t a)
//                 ^---Round to Integral exact, using current rounding mode
//                 (vector).
// float64x1_t vrndx_f64(float64x1_t a)
// float32x4_t vrndxq_f32(float32x4_t a)
// float64x2_t vrndxq_f64(float64x2_t a)

/* NOTE vrnd 使用 round-to-zero */
TEST_CASE(test_vrnd_f32) {
    static const struct {
        float a[2];
        float r[2];
    } test_vec[] = {
        {{-1.50, 1.50}, {-1.00, 1.00}},
        {{-2.50, 2.50}, {-2.00, 2.00}},
        {{782.33, 23.83}, {782.00, 23.00}},
        {{-231.98, -121.26}, {-231.00, -121.00}},
        {{524.61, 500.02}, {524.00, 500.00}},
        {{80.15, 517.44}, {80.00, 517.00}},
        {{-754.87, 128.37}, {-754.00, 128.00}},
        {{182.53, 136.96}, {182.00, 136.00}},
        {{605.41, -833.56}, {605.00, -833.00}},
        {{774.26, -578.69}, {774.00, -578.00}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t r = vrnd_f32(a);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}

TEST_CASE(test_vrndn_f32) {
    static const struct {
        float a[2];
        float r[2];
    } test_vec[] = {{{-1.50, 1.50}, {-2.00, 2.00}},
                    {{-2.50, 2.50}, {-2.00, 2.00}},
                    {{-593.90, 196.84}, {-594.00, 197.00}},
                    {{569.79, 336.27}, {570.00, 336.00}},
                    {{-670.11, 299.96}, {-670.00, 300.00}},
                    {{-4.27, -333.31}, {-4.00, -333.00}},
                    {{-389.20, 338.21}, {-389.00, 338.00}},
                    {{172.22, 764.71}, {172.00, 765.00}},
                    {{789.38, -740.62}, {789.00, -741.00}},
                    {{713.87, -75.96}, {714.00, -76.00}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t r = vrndn_f32(a);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}

TEST_CASE(test_vrndm_f32) {
    static const struct {
        float a[2];
        float r[2];
    } test_vec[] = {
        {{-1.50, 1.50}, {-2.00, 1.00}},
        {{-2.50, 2.50}, {-3.00, 2.00}},
        {{-897.30, 351.51}, {-898.00, 351.00}},
        {{-396.24, -136.90}, {-397.00, -137.00}},
        {{-966.64, 805.58}, {-967.00, 805.00}},
        {{848.81, -910.27}, {848.00, -911.00}},
        {{-262.75, 779.23}, {-263.00, 779.00}},
        {{824.19, -986.07}, {824.00, -987.00}},
        {{272.13, 812.56}, {272.00, 812.00}},
        {{-763.50, 477.59}, {-764.00, 477.00}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t r = vrndm_f32(a);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}
