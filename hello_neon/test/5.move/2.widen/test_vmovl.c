// 2023-04-20 15:05
#include <neon.h>
#include <neon_test.h>
// int16x8_t vmovl_s8(int8x8_t a)
// int32x4_t vmovl_s16(int16x4_t a)
// int64x2_t vmovl_s32(int32x2_t a)
// uint16x8_t vmovl_u8(uint8x8_t a)
// uint32x4_t vmovl_u16(uint16x4_t a)
// uint64x2_t vmovl_u32(uint32x2_t a)
//
// int16x8_t vmovl_high_s8(int8x16_t a)
// int32x4_t vmovl_high_s16(int16x8_t a)
// int64x2_t vmovl_high_s32(int32x4_t a)
// uint16x8_t vmovl_high_u8(uint8x16_t a)
// uint32x4_t vmovl_high_u16(uint16x8_t a)
// uint64x2_t vmovl_high_u32(uint32x4_t a)

TEST_CASE(test_vmovl_s8) {
    struct {
        int8_t a[8];
        int16_t r[8];
    } test_vec[] = {
        {{31, 71, 44, 91, -52, 8, 55, -52}, {31, 71, 44, 91, -52, 8, 55, -52}},
        {{65, -81, -57, 44, 26, -47, 67, -127},
         {65, -81, -57, 44, 26, -47, 67, -127}},
        {{98, 108, 40, 95, -117, -15, 121, 41},
         {98, 108, 40, 95, -117, -15, 121, 41}},
        {{-21, -77, -55, -69, 110, 126, 54, -115},
         {-21, -77, -55, -69, 110, 126, 54, -115}},
        {{-59, 98, -24, -110, 106, 31, 94, -85},
         {-59, 98, -24, -110, 106, 31, 94, -85}},
        {{-49, 38, -40, -23, -9, 27, 107, 90},
         {-49, 38, -40, -23, -9, 27, 107, 90}},
        {{-121, -109, -71, 19, -124, 50, 60, 112},
         {-121, -109, -71, 19, -124, 50, 60, 112}},
        {{-27, 5, 43, 83, -125, 98, -32, 72},
         {-27, 5, 43, 83, -125, 98, -32, 72}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int16x8_t r = vmovl_s8(a);
        int16x8_t check = vld1q_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vmovl_high_s8) {
    struct {
        int8_t a[16];
        int16_t r[8];
    } test_vec[] = {
        {{14, 54, -9, -31, -54, -22, 44, 62, 23, 107, -40, 21, -6, -80, 20,
          -87},
         {23, 107, -40, 21, -6, -80, 20, -87}},
        {{-84, 95, -109, -90, -111, 97, 80, -111, 69, 2, 28, -73, -74, 26, 48,
          -59},
         {69, 2, 28, -73, -74, 26, 48, -59}},
        {{80, 39, -90, 27, 18, -46, 89, 41, 62, 49, 63, 56, -30, 83, -30, -114},
         {62, 49, 63, 56, -30, 83, -30, -114}},
        {{-77, 117, 52, 68, -42, -124, -43, 28, -122, -15, -45, 61, 12, 3, 2,
          92},
         {-122, -15, -45, 61, 12, 3, 2, 92}},
        {{43, -88, 119, 61, 122, -47, 102, -72, 2, -91, -15, -28, -7, -45, 115,
          -84},
         {2, -91, -15, -28, -7, -45, 115, -84}},
        {{72, -89, -16, 31, 44, -59, 59, -78, -73, 14, -17, -61, 18, -15, 31,
          61},
         {-73, 14, -17, -61, 18, -15, 31, 61}},
        {{-103, -105, 122, 20, 104, -32, -52, 106, -122, -67, 79, INT8_MAX,
          -112, -62, 43, -39},
         {-122, -67, 79, 127, -112, -62, 43, -39}},
        {{105, 27, -8, -107, -32, 51, 72, -105, 65, 55, 90, 83, 41, 122, -112,
          -62},
         {65, 55, 90, 83, 41, 122, -112, -62}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x16_t a = vld1q_s8(test_vec[i].a);
        int16x8_t r = vmovl_high_s8(a);
        int16x8_t check = vld1q_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}