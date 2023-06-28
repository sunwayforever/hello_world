// 2023-04-20 17:35
#include <neon.h>
#include <neon_test.h>
// int8x8_t vorn_s8(int8x8_t a,int8x8_t b)
// int16x4_t vorn_s16(int16x4_t a,int16x4_t b)
// int32x2_t vorn_s32(int32x2_t a,int32x2_t b)
// int64x1_t vorn_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vorn_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vorn_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vorn_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vorn_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t vornq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vornq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vornq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vornq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vornq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vornq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vornq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vornq_u64(uint64x2_t a,uint64x2_t b)

TEST_CASE(test_vorn_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{71, -69, 126, -50, 51, -21, -47, -103},
         {30, 10, 57, 26, 126, -9, -8, 99},
         {-25, -1, -2, -17, -77, -21, -41, -99}},
        {{83, -92, 91, 51, 23, 56, 16, -37},
         {38, 14, -105, 93, 53, -77, -24, 124},
         {-37, -11, 123, -77, -33, 124, 23, -37}},
        {{111, 102, 74, -94, 82, 27, 59, 112},
         {38, 116, -118, -92, 107, -126, 7, -65},
         {-1, -17, INT8_MAX, -5, -42, INT8_MAX, -5, 112}},
        {{39, 99, -14, 62, -101, 2, 25, -63},
         {16, -80, 30, 69, 100, 6, -63, -45},
         {-17, 111, -13, -66, -101, -5, 63, -19}},
        {{108, 11, 117, -66, 39, -80, 46, 77},
         {36, -72, -15, -112, 59, -7, 79, 98},
         {-1, 79, INT8_MAX, -1, -25, -74, -66, -35}},
        {{92, 65, -96, -9, 67, -71, -72, 83},
         {105, -42, -103, -51, -36, 90, -96, 72},
         {-34, 105, -26, -9, 99, -67, -1, -9}},
        {{102, 21, 7, -115, -59, 53, -38, -22},
         {-18, -53, 122, 41, -60, -55, -117, 32},
         {119, 53, -121, -33, -1, 55, -2, -1}},
        {{10, 43, 23, 77, -28, -49, -96, 77},
         {-91, 57, 27, -127, -108, -69, -54, -6},
         {90, -17, -9, INT8_MAX, -17, -49, -75, 77}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vorn_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
