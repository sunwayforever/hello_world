// 2023-04-21 13:52
#include <neon.h>
#include <neon_test.h>
// vcobmine 没有 q 版本, 因为它的输出必然是 128-bit
// int8x16_t vcombine_s8(int8x8_t low,int8x8_t high)
// int16x8_t vcombine_s16(int16x4_t low,int16x4_t high)
// int32x4_t vcombine_s32(int32x2_t low,int32x2_t high)
// int64x2_t vcombine_s64(int64x1_t low,int64x1_t high)
// uint8x16_t vcombine_u8(uint8x8_t low,uint8x8_t high)
// uint16x8_t vcombine_u16(uint16x4_t low,uint16x4_t high)
// uint32x4_t vcombine_u32(uint32x2_t low,uint32x2_t high)
// uint64x2_t vcombine_u64(uint64x1_t low,uint64x1_t high)
// poly64x2_t vcombine_p64(poly64x1_t low,poly64x1_t high)
// float16x8_t vcombine_f16(float16x4_t low,float16x4_t high)
// float32x4_t vcombine_f32(float32x2_t low,float32x2_t high)
// poly8x16_t vcombine_p8(poly8x8_t low,poly8x8_t high)
// poly16x8_t vcombine_p16(poly16x4_t low,poly16x4_t high)
// float64x2_t vcombine_f64(float64x1_t low,float64x1_t high)

TEST_CASE(test_vcombine_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[16];
    } test_vec[] = {
        {{68, -50, -26, 105, 81, 69, 3, 21},
         {119, 72, 123, -44, 61, 43, 109, -34},
         {68, -50, -26, 105, 81, 69, 3, 21, 119, 72, 123, -44, 61, 43, 109,
          -34}},
        {{36, -100, 109, -72, -41, -75, 14, 114},
         {110, 126, -79, -75, 23, -2, -9, 91},
         {36, -100, 109, -72, -41, -75, 14, 114, 110, 126, -79, -75, 23, -2, -9,
          91}},
        {{-51, -34, -59, 30, 35, -56, 51, -102},
         {16, -82, 110, 77, -38, -37, 43, -2},
         {-51, -34, -59, 30, 35, -56, 51, -102, 16, -82, 110, 77, -38, -37, 43,
          -2}},
        {{119, -104, -74, 79, 78, -60, -63, -68},
         {66, 114, 113, 90, 112, 105, -75, 61},
         {119, -104, -74, 79, 78, -60, -63, -68, 66, 114, 113, 90, 112, 105,
          -75, 61}},
        {{71, 122, 91, 106, 67, -113, 4, 83},
         {61, 114, -95, 23, 77, -52, 22, -60},
         {71, 122, 91, 106, 67, -113, 4, 83, 61, 114, -95, 23, 77, -52, 22,
          -60}},
        {{101, -52, 19, -77, -111, -44, 111, -45},
         {70, -31, 45, -73, 74, -29, -12, -111},
         {101, -52, 19, -77, -111, -44, 111, -45, 70, -31, 45, -73, 74, -29,
          -12, -111}},
        {{93, 80, -5, -96, -33, -1, -12, 28},
         {113, -107, 52, -66, 97, 74, -126, -58},
         {93, 80, -5, -96, -33, -1, -12, 28, 113, -107, 52, -66, 97, 74, -126,
          -58}},
        {{22, -106, 121, -89, 106, -23, 123, -79},
         {-54, -88, 104, 20, -117, 92, -91, -23},
         {22, -106, 121, -89, 106, -23, 123, -79, -54, -88, 104, 20, -117, 92,
          -91, -23}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x16_t r = vcombine_s8(a, b);
        int8x16_t check = vld1q_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
