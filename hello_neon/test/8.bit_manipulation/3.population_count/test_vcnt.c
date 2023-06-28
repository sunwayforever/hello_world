// 2023-04-20 18:13
#include <neon.h>
#include <neon_test.h>
// int8x8_t vcnt_s8(int8x8_t a)
// uint8x8_t vcnt_u8(uint8x8_t a)
// poly8x8_t vcnt_p8(poly8x8_t a)
//
// int8x16_t vcntq_s8(int8x16_t a)
// uint8x16_t vcntq_u8(uint8x16_t a)
// poly8x16_t vcntq_p8(poly8x16_t a)

TEST_CASE(test_vcnt_s8) {
    struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{89, -96, -103, -70, -8, -10, 37, -116}, {4, 2, 4, 5, 5, 6, 3, 3}},
        {{-66, 58, -5, -94, 15, 122, 95, -35}, {6, 4, 7, 3, 4, 5, 6, 6}},
        {{-102, -68, 104, -83, -88, -36, -100, INT8_MIN},
         {4, 5, 3, 5, 3, 5, 4, 1}},
        {{-32, -127, 72, -57, -104, 77, 40, -15}, {3, 2, 2, 5, 3, 4, 2, 5}},
        {{-18, -63, -84, -26, -73, -47, 114, 117}, {6, 3, 4, 5, 6, 4, 4, 5}},
        {{11, 109, 23, 26, -25, 118, -9, -127}, {3, 5, 4, 3, 6, 5, 7, 2}},
        {{51, 96, 46, -37, 60, -54, 92, 28}, {4, 2, 4, 6, 4, 4, 4, 3}},
        {{76, -92, -28, -28, -15, 12, -43, -33}, {3, 3, 4, 4, 5, 2, 5, 7}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vcnt_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
