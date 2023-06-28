// 2023-04-21 16:16
#include <neon.h>
#include <neon_test.h>
// int8x8_t vtrn1_s8(int8x8_t a,int8x8_t b)
// int16x4_t vtrn1_s16(int16x4_t a,int16x4_t b)
// int32x2_t vtrn1_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vtrn1_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vtrn1_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vtrn1_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vtrn1_f32(float32x2_t a,float32x2_t b)
// poly8x8_t vtrn1_p8(poly8x8_t a,poly8x8_t b)
// poly16x4_t vtrn1_p16(poly16x4_t a,poly16x4_t b)
//
// int8x16_t vtrn1q_s8(int8x16_t a,int8x16_t b)
// int16x8_t vtrn1q_s16(int16x8_t a,int16x8_t b)
// int32x4_t vtrn1q_s32(int32x4_t a,int32x4_t b)
// int64x2_t vtrn1q_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vtrn1q_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vtrn1q_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vtrn1q_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vtrn1q_u64(uint64x2_t a,uint64x2_t b)
// poly64x2_t vtrn1q_p64(poly64x2_t a,poly64x2_t b)
// float32x4_t vtrn1q_f32(float32x4_t a,float32x4_t b)
// float64x2_t vtrn1q_f64(float64x2_t a,float64x2_t b)
// poly8x16_t vtrn1q_p8(poly8x16_t a,poly8x16_t b)
// poly16x8_t vtrn1q_p16(poly16x8_t a,poly16x8_t b)
// ---------------------------------------------------
// int8x8_t vtrn2_s8(int8x8_t a,int8x8_t b)
// int16x4_t vtrn2_s16(int16x4_t a,int16x4_t b)
// int32x2_t vtrn2_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vtrn2_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vtrn2_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vtrn2_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vtrn2_f32(float32x2_t a,float32x2_t b)
// poly8x8_t vtrn2_p8(poly8x8_t a,poly8x8_t b)
//
// int8x16_t vtrn2q_s8(int8x16_t a,int8x16_t b)
// int16x8_t vtrn2q_s16(int16x8_t a,int16x8_t b)
// int32x4_t vtrn2q_s32(int32x4_t a,int32x4_t b)
// int64x2_t vtrn2q_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vtrn2q_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vtrn2q_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vtrn2q_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vtrn2q_u64(uint64x2_t a,uint64x2_t b)
// poly64x2_t vtrn2q_p64(poly64x2_t a,poly64x2_t b)
// float32x4_t vtrn2q_f32(float32x4_t a,float32x4_t b)
// float64x2_t vtrn2q_f64(float64x2_t a,float64x2_t b)
// poly8x16_t vtrn2q_p8(poly8x16_t a,poly8x16_t b)
// poly16x4_t vtrn2_p16(poly16x4_t a,poly16x4_t b)
// poly16x8_t vtrn2q_p16(poly16x8_t a,poly16x8_t b)
// ---------------------------------------------------
// int8x8x2_t vtrn_s8(int8x8_t a,int8x8_t b)
// int16x4x2_t vtrn_s16(int16x4_t a,int16x4_t b)
// uint8x8x2_t vtrn_u8(uint8x8_t a,uint8x8_t b)
// uint16x4x2_t vtrn_u16(uint16x4_t a,uint16x4_t b)
// poly8x8x2_t vtrn_p8(poly8x8_t a,poly8x8_t b)
// poly16x4x2_t vtrn_p16(poly16x4_t a,poly16x4_t b)
// int32x2x2_t vtrn_s32(int32x2_t a,int32x2_t b)
// float32x2x2_t vtrn_f32(float32x2_t a,float32x2_t b)
// uint32x2x2_t vtrn_u32(uint32x2_t a,uint32x2_t b)
//
// int8x16x2_t vtrnq_s8(int8x16_t a,int8x16_t b)
// int16x8x2_t vtrnq_s16(int16x8_t a,int16x8_t b)
// int32x4x2_t vtrnq_s32(int32x4_t a,int32x4_t b)
// float32x4x2_t vtrnq_f32(float32x4_t a,float32x4_t b)
// uint8x16x2_t vtrnq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8x2_t vtrnq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4x2_t vtrnq_u32(uint32x4_t a,uint32x4_t b)
// poly8x16x2_t vtrnq_p8(poly8x16_t a,poly8x16_t b)
// poly16x8x2_t vtrnq_p16(poly16x8_t a,poly16x8_t b)

TEST_CASE(test_vtrn1_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-60, -45, -88, -68, -80, 32, 27, -59},
         {0, 78, -30, -82, -119, 46, 17, 49},
         {-60, 0, -88, -30, -80, -119, 27, 17}},
        {{101, 29, 88, 118, -106, -13, -99, 125},
         {71, -38, 116, -26, -54, -111, 13, -114},
         {101, 71, 88, 116, -106, -54, -99, 13}},
        {{101, -75, 75, 21, -43, 102, -37, -42},
         {-75, -67, -124, 62, -20, -107, 112, 81},
         {101, -75, 75, -124, -43, -20, -37, 112}},
        {{-77, -56, -57, 73, -69, 100, -58, 2},
         {62, 58, -23, 8, -52, -10, -105, 49},
         {-77, 62, -57, -23, -69, -52, -58, -105}},
        {{-84, -30, 70, -127, 72, 33, 87, -3},
         {-33, -37, 60, -53, 113, -84, 28, 36},
         {-84, -33, 70, 60, 72, 113, 87, 28}},
        {{116, -29, 109, 47, 71, 51, 50, -123},
         {110, 27, -115, 58, 17, 36, 107, -67},
         {116, 110, 109, -115, 71, 17, 50, 107}},
        {{6, -79, 63, 79, -45, -106, 76, -78},
         {114, -120, 125, -29, 52, -103, 7, -88},
         {6, 114, 63, 125, -45, 52, 76, 7}},
        {{124, 116, -40, -61, -89, 10, 72, 21},
         {37, -43, 79, 54, -6, -70, -12, 0},
         {124, 37, -40, 79, -89, -6, 72, -12}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vtrn1_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vtrn2_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-106, 93, 108, 69, 23, 43, 82, -47},
         {44, 36, -125, 77, 46, -121, 2, 4},
         {93, 36, 69, 77, 43, -121, -47, 4}},
        {{71, 37, -29, 29, -88, 1, 106, -41},
         {123, -54, -57, 117, -92, INT8_MIN, -68, 58},
         {37, -54, 29, 117, 1, INT8_MIN, -41, 58}},
        {{-35, 40, INT8_MIN, -11, 83, -46, -58, INT8_MAX},
         {-9, 73, -52, 37, -48, -49, 41, 24},
         {40, 73, -11, 37, -46, -49, INT8_MAX, 24}},
        {{-12, 13, 53, -100, 14, -96, 115, -118},
         {106, 59, -1, 14, -69, -69, 72, -104},
         {13, 59, -100, 14, -96, -69, -118, -104}},
        {{-29, -56, -115, 55, -101, 84, -74, -110},
         {-99, -125, -73, 110, 82, -31, -122, 70},
         {-56, -125, 55, 110, 84, -31, -110, 70}},
        {{-18, -69, -30, -4, 91, 85, -122, -59},
         {-112, -122, -45, 75, 65, 28, -28, 37},
         {-69, -122, -4, 75, 85, 28, -59, 37}},
        {{-28, 113, 92, INT8_MAX, -59, 18, 17, 99},
         {-107, -55, -47, -25, -86, 87, 45, -104},
         {113, -55, INT8_MAX, -25, 18, 87, 99, -104}},
        {{18, 15, -108, 110, 101, 27, 51, -11},
         {-95, 7, 65, -30, 35, 37, 7, 7},
         {15, 7, 110, -30, 27, 37, -11, 7}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vtrn2_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
