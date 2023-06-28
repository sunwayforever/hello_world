// 2023-04-18 11:41
#include <neon.h>
#include <neon_test.h>

// int8x8_t vpadd_s8(int8x8_t a,int8x8_t b)
//          ^---pair
// int16x4_t vpadd_s16(int16x4_t a,int16x4_t b)
// int32x2_t vpadd_s32(int32x2_t a,int32x2_t b)
// uint8x8_t vpadd_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vpadd_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vpadd_u32(uint32x2_t a,uint32x2_t b)
// float32x2_t vpadd_f32(float32x2_t a,float32x2_t b)
//
// int8x16_t vpaddq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vpaddq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vpaddq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vpaddq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vpaddq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vpaddq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vpaddq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vpaddq_u64(uint64x2_t a,uint64x2_t b)
// float32x4_t vpaddq_f32(float32x4_t a,float32x4_t b)
// float64x2_t vpaddq_f64(float64x2_t a,float64x2_t b)
// ----------------------------------------------------
// int64_t vpaddd_s64(int64x2_t a)
//              ^---scalar
// uint64_t vpaddd_u64(uint64x2_t a)
// float32_t vpadds_f32(float32x2_t a)
// float64_t vpaddd_f64(float64x2_t a)

TEST_CASE(test_simde_vpadd_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-42, -32, -64, 117, 61, 45, -26, 96},
         {-49, 76, -77, -78, 47, -116, 78, -108},
         {-74, 53, 106, 70, 27, 101, -69, -30}},
        {{3, -93, -22, 89, 69, 121, -64, 110},
         {-10, -63, -118, 71, 28, -42, -14, -14},
         {-90, 67, -66, 46, -73, -47, -14, -28}},
        {{-74, -78, 103, -13, -33, 77, 83, -82},
         {-103, 6, 96, -56, -110, -82, 92, -106},
         {104, 90, 44, 1, -97, 40, 64, -14}},
        {{82, 71, -17, -105, -64, -80, 5, -74},
         {113, -112, -2, -115, 102, -16, INT8_MAX, 28},
         {-103, -122, 112, -69, 1, -117, 86, -101}},
        {{-94, -26, 15, -127, 51, 98, 47, -52},
         {104, -112, -108, -6, 62, -15, -112, -112},
         {-120, -112, -107, -5, -8, -114, 47, 32}},
        {{56, INT8_MIN, 39, -8, 48, 45, -81, -95},
         {-67, -83, 47, 35, -99, -82, 63, 63},
         {-72, 31, 93, 80, 106, 82, 75, 126}},
        {{-107, 78, -64, -56, -80, -17, -107, 24},
         {INT8_MAX, 41, 18, -66, 26, -93, 78, 82},
         {-29, -120, -97, -83, -88, -48, -67, -96}},
        {{35, 118, 75, 83, -93, -6, -12, 96},
         {-89, 35, -125, 68, -46, -62, -125, 103},
         {-103, -98, -99, 84, -54, -57, -108, -22}},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vpadd_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
