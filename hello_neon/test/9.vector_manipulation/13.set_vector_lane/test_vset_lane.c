// 2023-04-21 16:30
#include <neon.h>
#include <neon_test.h>
// uint8x8_t vset_lane_u8(uint8_t a,uint8x8_t v,const int lane)
// uint16x4_t vset_lane_u16(uint16_t a,uint16x4_t v,const int lane)
// uint32x2_t vset_lane_u32(uint32_t a,uint32x2_t v,const int lane)
// uint64x1_t vset_lane_u64(uint64_t a,uint64x1_t v,const int lane)
// poly64x1_t vset_lane_p64(poly64_t a,poly64x1_t v,const int lane)
// int8x8_t vset_lane_s8(int8_t a,int8x8_t v,const int lane)
// int16x4_t vset_lane_s16(int16_t a,int16x4_t v,const int lane)
// int32x2_t vset_lane_s32(int32_t a,int32x2_t v,const int lane)
// int64x1_t vset_lane_s64(int64_t a,int64x1_t v,const int lane)
// poly8x8_t vset_lane_p8(poly8_t a,poly8x8_t v,const int lane)
// poly16x4_t vset_lane_p16(poly16_t a,poly16x4_t v,const int lane)
// float16x4_t vset_lane_f16(float16_t a,float16x4_t v,const int lane)
// float16x8_t vsetq_lane_f16(float16_t a,float16x8_t v,const int lane)
// float32x2_t vset_lane_f32(float32_t a,float32x2_t v,const int lane)
// float64x1_t vset_lane_f64(float64_t a,float64x1_t v,const int lane)
//
// uint8x16_t vsetq_lane_u8(uint8_t a,uint8x16_t v,const int lane)
// uint16x8_t vsetq_lane_u16(uint16_t a,uint16x8_t v,const int lane)
// uint32x4_t vsetq_lane_u32(uint32_t a,uint32x4_t v,const int lane)
// uint64x2_t vsetq_lane_u64(uint64_t a,uint64x2_t v,const int lane)
// poly64x2_t vsetq_lane_p64(poly64_t a,poly64x2_t v,const int lane)
// int8x16_t vsetq_lane_s8(int8_t a,int8x16_t v,const int lane)
// int16x8_t vsetq_lane_s16(int16_t a,int16x8_t v,const int lane)
// int32x4_t vsetq_lane_s32(int32_t a,int32x4_t v,const int lane)
// int64x2_t vsetq_lane_s64(int64_t a,int64x2_t v,const int lane)
// poly8x16_t vsetq_lane_p8(poly8_t a,poly8x16_t v,const int lane)
// poly16x8_t vsetq_lane_p16(poly16_t a,poly16x8_t v,const int lane)
// float32x4_t vsetq_lane_f32(float32_t a,float32x4_t v,const int lane)
// float64x2_t vsetq_lane_f64(float64_t a,float64x2_t v,const int lane)

TEST_CASE(test_vset_lane_s8) {
    static const struct {
        int8_t a;
        int8_t v[8];
        int8_t r[8];
    } test_vec[] = {
        {70,
         {-27, 121, -72, -82, 88, -125, -103, 16},
         {70, 121, -72, -82, 88, -125, -103, 16}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8_t a = test_vec[i].a;
        int8x8_t v = vld1_s8(test_vec[i].v);
        int8x8_t r = vset_lane_s8(a, v, 0);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
