// 2023-04-21 11:34
#include <neon.h>
#include <neon_test.h>
// int8x8_t vdup_n_s8(int8_t value)
// int16x4_t vdup_n_s16(int16_t value)
// int32x2_t vdup_n_s32(int32_t value)
// int64x1_t vdup_n_s64(int64_t value)
// uint8x8_t vdup_n_u8(uint8_t value)
// uint16x4_t vdup_n_u16(uint16_t value)
// uint32x2_t vdup_n_u32(uint32_t value)
// uint64x1_t vdup_n_u64(uint64_t value)
// poly64x1_t vdup_n_p64(poly64_t value)
// float32x2_t vdup_n_f32(float32_t value)
// poly8x8_t vdup_n_p8(poly8_t value)
// poly16x4_t vdup_n_p16(poly16_t value)
// float64x1_t vdup_n_f64(float64_t value)
//
// int8x16_t vdupq_n_s8(int8_t value)
// int16x8_t vdupq_n_s16(int16_t value)
// int32x4_t vdupq_n_s32(int32_t value)
// int64x2_t vdupq_n_s64(int64_t value)
// uint8x16_t vdupq_n_u8(uint8_t value)
// uint16x8_t vdupq_n_u16(uint16_t value)
// uint32x4_t vdupq_n_u32(uint32_t value)
// uint64x2_t vdupq_n_u64(uint64_t value)
// poly64x2_t vdupq_n_p64(poly64_t value)
// float32x4_t vdupq_n_f32(float32_t value)
// poly8x16_t vdupq_n_p8(poly8_t value)
// poly16x8_t vdupq_n_p16(poly16_t value)
// float64x2_t vdupq_n_f64(float64_t value)
// ----------------------------------------
// int8x8_t vmov_n_s8(int8_t value)
//           ^^^ vmov_n 与 vdump_n 完全相同
// int16x4_t vmov_n_s16(int16_t value)
// int32x2_t vmov_n_s32(int32_t value)
// int64x1_t vmov_n_s64(int64_t value)
// uint8x8_t vmov_n_u8(uint8_t value)
// uint16x4_t vmov_n_u16(uint16_t value)
// uint32x2_t vmov_n_u32(uint32_t value)
// uint64x1_t vmov_n_u64(uint64_t value)
// float32x2_t vmov_n_f32(float32_t value)
// poly8x8_t vmov_n_p8(poly8_t value)
// poly16x4_t vmov_n_p16(poly16_t value)
// float64x1_t vmov_n_f64(float64_t value)
//
// int8x16_t vmovq_n_s8(int8_t value)
// int16x8_t vmovq_n_s16(int16_t value)
// int32x4_t vmovq_n_s32(int32_t value)
// int64x2_t vmovq_n_s64(int64_t value)
// uint8x16_t vmovq_n_u8(uint8_t value)
// uint16x8_t vmovq_n_u16(uint16_t value)
// uint32x4_t vmovq_n_u32(uint32_t value)
// uint64x2_t vmovq_n_u64(uint64_t value)
// float32x4_t vmovq_n_f32(float32_t value)
// poly8x16_t vmovq_n_p8(poly8_t value)
// poly16x8_t vmovq_n_p16(poly16_t value)
// float64x2_t vmovq_n_f64(float64_t value)
// ----------------------------------------------------
// int8x8_t vdup_lane_s8(int8x8_t vec,const int lane)
// int16x4_t vdup_lane_s16(int16x4_t vec,const int lane)
// int32x2_t vdup_lane_s32(int32x2_t vec,const int lane)
// int64x1_t vdup_lane_s64(int64x1_t vec,const int lane)
// uint8x8_t vdup_lane_u8(uint8x8_t vec,const int lane)
// uint16x4_t vdup_lane_u16(uint16x4_t vec,const int lane)
// uint32x2_t vdup_lane_u32(uint32x2_t vec,const int lane)
// uint64x1_t vdup_lane_u64(uint64x1_t vec,const int lane)
// poly64x1_t vdup_lane_p64(poly64x1_t vec,const int lane)
// float32x2_t vdup_lane_f32(float32x2_t vec,const int lane)
// poly8x8_t vdup_lane_p8(poly8x8_t vec,const int lane)
// poly16x4_t vdup_lane_p16(poly16x4_t vec,const int lane)
// float64x1_t vdup_lane_f64(float64x1_t vec,const int lane)
//
// int8x16_t vdupq_lane_s8(int8x8_t vec,const int lane)
// int16x8_t vdupq_lane_s16(int16x4_t vec,const int lane)
// int32x4_t vdupq_lane_s32(int32x2_t vec,const int lane)
// int64x2_t vdupq_lane_s64(int64x1_t vec,const int lane)
// uint8x16_t vdupq_lane_u8(uint8x8_t vec,const int lane)
// uint16x8_t vdupq_lane_u16(uint16x4_t vec,const int lane)
// uint32x4_t vdupq_lane_u32(uint32x2_t vec,const int lane)
// uint64x2_t vdupq_lane_u64(uint64x1_t vec,const int lane)
// poly64x2_t vdupq_lane_p64(poly64x1_t vec,const int lane)
// float32x4_t vdupq_lane_f32(float32x2_t vec,const int lane)
// poly8x16_t vdupq_lane_p8(poly8x8_t vec,const int lane)
// poly16x8_t vdupq_lane_p16(poly16x4_t vec,const int lane)
// float64x2_t vdupq_lane_f64(float64x1_t vec,const int lane)
// -----------------------------------------------------------
// int8x8_t vdup_laneq_s8(int8x16_t vec,const int lane)
// int16x4_t vdup_laneq_s16(int16x8_t vec,const int lane)
// int32x2_t vdup_laneq_s32(int32x4_t vec,const int lane)
// int64x1_t vdup_laneq_s64(int64x2_t vec,const int lane)
// uint8x8_t vdup_laneq_u8(uint8x16_t vec,const int lane)
// uint16x4_t vdup_laneq_u16(uint16x8_t vec,const int lane)
// uint32x2_t vdup_laneq_u32(uint32x4_t vec,const int lane)
// uint64x1_t vdup_laneq_u64(uint64x2_t vec,const int lane)
// poly64x1_t vdup_laneq_p64(poly64x2_t vec,const int lane)
// float32x2_t vdup_laneq_f32(float32x4_t vec,const int lane)
// poly8x8_t vdup_laneq_p8(poly8x16_t vec,const int lane)
// poly16x4_t vdup_laneq_p16(poly16x8_t vec,const int lane)
// float64x1_t vdup_laneq_f64(float64x2_t vec,const int lane)
//
// int8x16_t vdupq_laneq_s8(int8x16_t vec,const int lane)
// int16x8_t vdupq_laneq_s16(int16x8_t vec,const int lane)
// int32x4_t vdupq_laneq_s32(int32x4_t vec,const int lane)
// int64x2_t vdupq_laneq_s64(int64x2_t vec,const int lane)
// uint8x16_t vdupq_laneq_u8(uint8x16_t vec,const int lane)
// uint16x8_t vdupq_laneq_u16(uint16x8_t vec,const int lane)
// uint32x4_t vdupq_laneq_u32(uint32x4_t vec,const int lane)
// uint64x2_t vdupq_laneq_u64(uint64x2_t vec,const int lane)
// poly64x2_t vdupq_laneq_p64(poly64x2_t vec,const int lane)
// float32x4_t vdupq_laneq_f32(float32x4_t vec,const int lane)
// poly8x16_t vdupq_laneq_p8(poly8x16_t vec,const int lane)
// poly16x8_t vdupq_laneq_p16(poly16x8_t vec,const int lane)
// float64x2_t vdupq_laneq_f64(float64x2_t vec,const int lane)

TEST_CASE(test_simde_vdup_n_s8) {
    struct {
        int8_t a;
        int8_t r[8];
    } test_vec[] = {
        {-125, {-125, -125, -125, -125, -125, -125, -125, -125}},
        {51, {51, 51, 51, 51, 51, 51, 51, 51}},
        {-121, {-121, -121, -121, -121, -121, -121, -121, -121}},
        {-82, {-82, -82, -82, -82, -82, -82, -82, -82}},
        {-27, {-27, -27, -27, -27, -27, -27, -27, -27}},
        {-6, {-6, -6, -6, -6, -6, -6, -6, -6}},
        {-22, {-22, -22, -22, -22, -22, -22, -22, -22}},
        {103, {103, 103, 103, 103, 103, 103, 103, 103}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t r = vdup_n_s8(test_vec[i].a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
