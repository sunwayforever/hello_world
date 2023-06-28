// 2023-04-21 10:13
#include <neon.h>
#include <neon_test.h>
// int8x8_t vrbit_s8(int8x8_t a)
// uint8x8_t vrbit_u8(uint8x8_t a)
// poly8x8_t vrbit_p8(poly8x8_t a)
//
// int8x16_t vrbitq_s8(int8x16_t a)
// uint8x16_t vrbitq_u8(uint8x16_t a)
// poly8x16_t vrbitq_p8(poly8x16_t a)

TEST_CASE(test_vrbit_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{-66, 67, 108, 50, -14, 85, -13, 64},
         {125, -62, 54, 76, 79, -86, -49, 2}},
        {{7, 91, -13, 90, -66, 87, 56, 33},
         {-32, -38, -49, 90, 125, -22, 28, -124}},
        {{-121, 54, -123, 5, 65, 58, -81, -45},
         {-31, 108, -95, -96, -126, 92, -11, -53}},
        {{-31, -67, -125, -9, 93, -67, -11, 27},
         {-121, -67, -63, -17, -70, -67, -81, -40}},
        {{0, 97, 78, -14, -74, 65, 50, -67},
         {0, -122, 114, 79, 109, -126, 76, -67}},
        {{-99, 37, 23, 91, 125, 79, 124, 4},
         {-71, -92, -24, -38, -66, -14, 62, 32}},
        {{-123, 2, 10, -57, 60, -71, -102, 29},
         {-95, 64, 80, -29, 60, -99, 89, -72}},
        {{118, 30, 21, -45, -37, 10, -17, -37},
         {110, 120, -88, -53, -37, 80, -9, -37}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vrbit_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
