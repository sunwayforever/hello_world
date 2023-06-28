// 2023-04-17 14:40
#include <neon.h>
#include <neon_test.h>
// clang format off
// poly8x8_t vmul_p8(poly8x8_t a,poly8x8_t b)
// poly8x16_t vmulq_p8(poly8x16_t a,poly8x16_t b)
//                ^---128-bit vector
// poly16x8_t vmull_p8(poly8x8_t a,poly8x8_t b)
//                ^---widen
// poly16x8_t vmull_high_p8(poly8x16_t a,poly8x16_t b)
//                  ^^^^---high
// clang format on
TEST_CASE(test_vmll_p8) {
    struct {
        uint8_t a[8];
        uint8_t b[8];
        uint8_t r[8];
    } test_vec[] = {
        {{3, 59, 38, 92, 101, 69, 33, 125},
         {3, 9, 111, 27, 82, 53, 72, 89},
         {5, 227, 130, 196, 218, 161, 72, 5}},
        {{49, 114, 110, 77, 44, 114, 19, 70},
         {124, 116, 15, 94, 118, 102, 35, 31},
         {60, 40, 122, 86, 168, 236, 85, 130}},
        {{95, 73, 61, 7, 4, 94, 118, 120},
         {104, 6, 93, 22, 59, 21, 67, 108},
         {216, 182, 177, 98, 236, 198, 26, 32}},
        {{121, 80, 71, 76, 34, 52, 110, 99},
         {88, 96, 5, 31, 58, 40, 0, 104},
         {24, 0, 91, 68, 52, 32, 0, 184}},
        {{113, 61, 111, 117, 101, 7, 19, 3},
         {13, 75, 25, 72, 96, 41, 76, 24},
         {61, 239, 231, 232, 224, 223, 20, 40}},
        {{39, 109, 100, 5, 56, 10, 104, 112},
         {106, 109, 113, 49, 107, 113, 103, 6},
         {86, 81, 164, 245, 136, 106, 24, 32}},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint8x8_t a = vld1_u8(test_vec[i].a);
        uint8x8_t b = vld1_u8(test_vec[i].b);
        poly8x8_t _a = vreinterpret_p8_u8(a);
        poly8x8_t _b = vreinterpret_p8_u8(b);
        poly8x8_t r = vmul_p8(_a, _b);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
}
