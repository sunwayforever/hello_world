// 2023-04-20 19:27
#include <neon.h>
#include <neon_test.h>
// int8x8_t vcopy_lane_s8(int8x8_t a,const int lane1,int8x8_t b,const int lane2)
// int16x4_t vcopy_lane_s16(int16x4_t a,const int lane1,int16x4_t b,const int
// lane2) int32x2_t vcopy_lane_s32(int32x2_t a,const int lane1,int32x2_t b,const
// int lane2) int64x1_t vcopy_lane_s64(int64x1_t a,const int lane1,int64x1_t
// b,const int lane2) uint8x8_t vcopy_lane_u8(uint8x8_t a,const int
// lane1,uint8x8_t b,const int lane2) uint16x4_t vcopy_lane_u16(uint16x4_t
// a,const int lane1,uint16x4_t b,const int lane2) uint32x2_t
// vcopy_lane_u32(uint32x2_t a,const int lane1,uint32x2_t b,const int lane2)
// uint64x1_t vcopy_lane_u64(uint64x1_t a,const int lane1,uint64x1_t b,const int
// lane2) poly64x1_t vcopy_lane_p64(poly64x1_t a,const int lane1,poly64x1_t
// b,const int lane2) float32x2_t vcopy_lane_f32(float32x2_t a,const int
// lane1,float32x2_t b,const int lane2) float64x1_t vcopy_lane_f64(float64x1_t
// a,const int lane1,float64x1_t b,const int lane2) poly8x8_t
// vcopy_lane_p8(poly8x8_t a,const int lane1,poly8x8_t b,const int lane2)
// poly16x4_t vcopy_lane_p16(poly16x4_t a,const int lane1,poly16x4_t b,const int
// lane2)
//
// int8x16_t vcopyq_lane_s8(int8x16_t a,const int lane1,int8x8_t b,const int
// lane2) int16x8_t vcopyq_lane_s16(int16x8_t a,const int lane1,int16x4_t
// b,const int lane2) int32x4_t vcopyq_lane_s32(int32x4_t a,const int
// lane1,int32x2_t b,const int lane2) int64x2_t vcopyq_lane_s64(int64x2_t
// a,const int lane1,int64x1_t b,const int lane2) uint8x16_t
// vcopyq_lane_u8(uint8x16_t a,const int lane1,uint8x8_t b,const int lane2)
// uint16x8_t vcopyq_lane_u16(uint16x8_t a,const int lane1,uint16x4_t b,const
// int lane2) uint32x4_t vcopyq_lane_u32(uint32x4_t a,const int lane1,uint32x2_t
// b,const int lane2) uint64x2_t vcopyq_lane_u64(uint64x2_t a,const int
// lane1,uint64x1_t b,const int lane2) poly64x2_t vcopyq_lane_p64(poly64x2_t
// a,const int lane1,poly64x1_t b,const int lane2) float32x4_t
// vcopyq_lane_f32(float32x4_t a,const int lane1,float32x2_t b,const int lane2)
// float64x2_t vcopyq_lane_f64(float64x2_t a,const int lane1,float64x1_t b,const
// int lane2) poly8x16_t vcopyq_lane_p8(poly8x16_t a,const int lane1,poly8x8_t
// b,const int lane2) poly16x8_t vcopyq_lane_p16(poly16x8_t a,const int
// lane1,poly16x4_t b,const int lane2)
// ----------------------------------------------------------------------------------------
// int8x8_t vcopy_laneq_s8(int8x8_t a,const int lane1,int8x16_t b,const int
// lane2) int16x4_t vcopy_laneq_s16(int16x4_t a,const int lane1,int16x8_t
// b,const int lane2) int32x2_t vcopy_laneq_s32(int32x2_t a,const int
// lane1,int32x4_t b,const int lane2) int64x1_t vcopy_laneq_s64(int64x1_t
// a,const int lane1,int64x2_t b,const int lane2) uint8x8_t
// vcopy_laneq_u8(uint8x8_t a,const int lane1,uint8x16_t b,const int lane2)
// uint16x4_t vcopy_laneq_u16(uint16x4_t a,const int lane1,uint16x8_t b,const
// int lane2) uint32x2_t vcopy_laneq_u32(uint32x2_t a,const int lane1,uint32x4_t
// b,const int lane2) uint64x1_t vcopy_laneq_u64(uint64x1_t a,const int
// lane1,uint64x2_t b,const int lane2) poly64x1_t vcopy_laneq_p64(poly64x1_t
// a,const int lane1,poly64x2_t b,const int lane2) float32x2_t
// vcopy_laneq_f32(float32x2_t a,const int lane1,float32x4_t b,const int lane2)
// float64x1_t vcopy_laneq_f64(float64x1_t a,const int lane1,float64x2_t b,const
// int lane2) poly8x8_t vcopy_laneq_p8(poly8x8_t a,const int lane1,poly8x16_t
// b,const int lane2) poly16x4_t vcopy_laneq_p16(poly16x4_t a,const int
// lane1,poly16x8_t b,const int lane2)
//
// int8x16_t vcopyq_laneq_s8(int8x16_t a,const int lane1,int8x16_t b,const int
// lane2) int16x8_t vcopyq_laneq_s16(int16x8_t a,const int lane1,int16x8_t
// b,const int lane2) int32x4_t vcopyq_laneq_s32(int32x4_t a,const int
// lane1,int32x4_t b,const int lane2) int64x2_t vcopyq_laneq_s64(int64x2_t
// a,const int lane1,int64x2_t b,const int lane2) uint8x16_t
// vcopyq_laneq_u8(uint8x16_t a,const int lane1,uint8x16_t b,const int lane2)
// uint16x8_t vcopyq_laneq_u16(uint16x8_t a,const int lane1,uint16x8_t b,const
// int lane2) uint32x4_t vcopyq_laneq_u32(uint32x4_t a,const int
// lane1,uint32x4_t b,const int lane2) uint64x2_t vcopyq_laneq_u64(uint64x2_t
// a,const int lane1,uint64x2_t b,const int lane2) poly64x2_t
// vcopyq_laneq_p64(poly64x2_t a,const int lane1,poly64x2_t b,const int lane2)
// float32x4_t vcopyq_laneq_f32(float32x4_t a,const int lane1,float32x4_t
// b,const int lane2) float64x2_t vcopyq_laneq_f64(float64x2_t a,const int
// lane1,float64x2_t b,const int lane2) poly8x16_t vcopyq_laneq_p8(poly8x16_t
// a,const int lane1,poly8x16_t b,const int lane2) poly16x8_t
// vcopyq_laneq_p16(poly16x8_t a,const int lane1,poly16x8_t b,const int lane2)

TEST_CASE(test_vcopy_lane_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{121, -36, -63, 101, 97, -69, 84, 14},
         {-55, 67, -91, 27, -125, -19, -21, -34},
         {121, 67, -63, 101, 97, -69, 84, 14}},
        {{28, -95, -79, -108, -88, 3, -30, 34},
         {-33, -92, -121, 64, 95, -37, 79, 40},
         {28, -92, -79, -108, -88, 3, -30, 34}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vcopy_lane_s8(a, 1, b, 1);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
