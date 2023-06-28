// 2023-04-21 14:29
#include <neon.h>
#include <neon_test.h>
// int8x8_t vext_s8(int8x8_t a,int8x8_t b,const int n)
// int16x4_t vext_s16(int16x4_t a,int16x4_t b,const int n)
// int32x2_t vext_s32(int32x2_t a,int32x2_t b,const int n)
// int64x1_t vext_s64(int64x1_t a,int64x1_t b,const int n)
// uint8x8_t vext_u8(uint8x8_t a,uint8x8_t b,const int n)
// uint16x4_t vext_u16(uint16x4_t a,uint16x4_t b,const int n)
// uint32x2_t vext_u32(uint32x2_t a,uint32x2_t b,const int n)
// uint64x1_t vext_u64(uint64x1_t a,uint64x1_t b,const int n)
// poly64x1_t vext_p64(poly64x1_t a,poly64x1_t b,const int n)
// float32x2_t vext_f32(float32x2_t a,float32x2_t b,const int n)
// float64x1_t vext_f64(float64x1_t a,float64x1_t b,const int n)
// poly8x8_t vext_p8(poly8x8_t a,poly8x8_t b,const int n)
// poly16x4_t vext_p16(poly16x4_t a,poly16x4_t b,const int n)
//
// int8x16_t vextq_s8(int8x16_t a,int8x16_t b,const int n)
// int16x8_t vextq_s16(int16x8_t a,int16x8_t b,const int n)
// int32x4_t vextq_s32(int32x4_t a,int32x4_t b,const int n)
// int64x2_t vextq_s64(int64x2_t a,int64x2_t b,const int n)
// uint8x16_t vextq_u8(uint8x16_t a,uint8x16_t b,const int n)
// uint16x8_t vextq_u16(uint16x8_t a,uint16x8_t b,const int n)
// uint32x4_t vextq_u32(uint32x4_t a,uint32x4_t b,const int n)
// uint64x2_t vextq_u64(uint64x2_t a,uint64x2_t b,const int n)
// poly64x2_t vextq_p64(poly64x2_t a,poly64x2_t b,const int n)
// float32x4_t vextq_f32(float32x4_t a,float32x4_t b,const int n)
// float64x2_t vextq_f64(float64x2_t a,float64x2_t b,const int n)
// poly8x16_t vextq_p8(poly8x16_t a,poly8x16_t b,const int n)
// poly16x8_t vextq_p16(poly16x8_t a,poly16x8_t b,const int n)

TEST_CASE(test_vext_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{76, 122, -21, -77, -52, 88, -109, -51},
         {-91, 82, -77, -45, 24, -52, -41, -73},
         {122, -21, -77, -52, 88, -109, -51, -91}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vext_s8(a, b, 1);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
