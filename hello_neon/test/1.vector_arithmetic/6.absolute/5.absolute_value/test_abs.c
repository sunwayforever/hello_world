// 2023-04-17 16:36
#include <neon.h>
#include <neon_test.h>
// int8x8_t vabs_s8(int8x8_t a)
// int16x4_t vabs_s16(int16x4_t a)
// int32x2_t vabs_s32(int32x2_t a)
// int64x1_t vabs_s64(int64x1_t a)
// float32x2_t vabs_f32(float32x2_t a)
// float64x1_t vabs_f64(float64x1_t a)
// -----------------------------------
// int8x16_t vabsq_s8(int8x16_t a)
// int16x8_t vabsq_s16(int16x8_t a)
// int32x4_t vabsq_s32(int32x4_t a)
// int64x2_t vabsq_s64(int64x2_t a)
// float32x4_t vabsq_f32(float32x4_t a)
// float64x2_t vabsq_f64(float64x2_t a)
// ------------------------------------
// int64_t vabsd_s64(int64_t a)
//             ^---DI

TEST_CASE(test_vabs_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{INT8_MIN, -18, 117, -22, 13, -62, -94, 57},
         {INT8_MIN, 18, 117, 22, 13, 62, 94, 57}},
        {{32, 12, INT8_MIN, 3, -50, 38, -120, 34},
         {32, 12, INT8_MIN, 3, 50, 38, 120, 34}},
        {{7, 100, -64, -52, -66, -82, -16, 44},
         {7, 100, 64, 52, 66, 82, 16, 44}},
        {{62, -64, 55, 87, -99, 82, -13, -62},
         {62, 64, 55, 87, 99, 82, 13, 62}},
        {{64, 105, -84, 77, 43, 78, -121, 75},
         {64, 105, 84, 77, 43, 78, 121, 75}},
        {{90, 7, 78, 40, 45, -41, 75, 52}, {90, 7, 78, 40, 45, 41, 75, 52}},
        {{59, 11, 0, -7, -71, -15, 38, -8}, {59, 11, 0, 7, 71, 15, 38, 8}},
        {{-79, 93, 79, 78, -81, 67, 16, -16},
         {79, 93, 79, 78, 81, 67, 16, 16}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vabs_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }

    return 0;
}
