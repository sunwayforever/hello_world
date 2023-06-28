// 2023-04-20 16:58
#include <neon.h>
#include <neon_test.h>
// int8x8_t vmvn_s8(int8x8_t a)
// int16x4_t vmvn_s16(int16x4_t a)
// int32x2_t vmvn_s32(int32x2_t a)
// uint8x8_t vmvn_u8(uint8x8_t a)
// uint16x4_t vmvn_u16(uint16x4_t a)
// uint32x2_t vmvn_u32(uint32x2_t a)
// poly8x8_t vmvn_p8(poly8x8_t a)
//
// int8x16_t vmvnq_s8(int8x16_t a)
// int16x8_t vmvnq_s16(int16x8_t a)
// int32x4_t vmvnq_s32(int32x4_t a)
// uint8x16_t vmvnq_u8(uint8x16_t a)
// uint16x8_t vmvnq_u16(uint16x8_t a)
// uint32x4_t vmvnq_u32(uint32x4_t a)
// poly8x16_t vmvnq_p8(poly8x16_t a)

TEST_CASE(test_vmvn_s8) {
    static const struct {
        int8_t a[8];
        int8_t r[8];
    } test_vec[] = {
        {{27, 6, 75, 101, 8, -12, -2, 29},
         {-28, -7, -76, -102, -9, 11, 1, -30}},
        {{81, -96, -60, 86, -57, 42, -34, -29},
         {-82, 95, 59, -87, 56, -43, 33, 28}},
        {{120, 82, -78, 74, -126, 123, 26, 61},
         {-121, -83, 77, -75, 125, -124, -27, -62}},
        {{-113, 122, -79, -94, -16, 93, -97, 11},
         {112, -123, 78, 93, 15, -94, 96, -12}},
        {{99, -21, 112, 107, -33, 111, -120, 48},
         {-100, 20, -113, -108, 32, -112, 119, -49}},
        {{15, 76, -122, -42, 118, 100, -71, -17},
         {-16, -77, 121, 41, -119, -101, 70, 16}},
        {{-74, 107, 57, 57, -26, 83, 118, 117},
         {73, -108, -58, -58, 25, -84, -119, -118}},
        {{-51, 39, 23, -67, -124, -73, -55, -25},
         {50, -40, -24, 66, 123, 72, 54, 24}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t r = vmvn_s8(a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
