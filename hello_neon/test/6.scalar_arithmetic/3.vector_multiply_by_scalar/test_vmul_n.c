// 2023-04-20 16:21
#include <neon.h>
#include <neon_test.h>
// int16x4_t vmul_n_s16(int16x4_t a,int16_t b)
// int32x2_t vmul_n_s32(int32x2_t a,int32_t b)
// uint16x4_t vmul_n_u16(uint16x4_t a,uint16_t b)
// uint32x2_t vmul_n_u32(uint32x2_t a,uint32_t b)
// float32x2_t vmul_n_f32(float32x2_t a,float32_t b)
// float64x1_t vmul_n_f64(float64x1_t a,float64_t b)
//
// int16x8_t vmulq_n_s16(int16x8_t a,int16_t b)
// int32x4_t vmulq_n_s32(int32x4_t a,int32_t b)
// uint16x8_t vmulq_n_u16(uint16x8_t a,uint16_t b)
// uint32x4_t vmulq_n_u32(uint32x4_t a,uint32_t b)
// float32x4_t vmulq_n_f32(float32x4_t a,float32_t b)
// float64x2_t vmulq_n_f64(float64x2_t a,float64_t b)
// ---------------------------------------------------------------
// int16x4_t vmul_lane_s16(int16x4_t a,int16x4_t v,const int lane)
// int32x2_t vmul_lane_s32(int32x2_t a,int32x2_t v,const int lane)
// uint16x4_t vmul_lane_u16(uint16x4_t a,uint16x4_t v,const int lane)
// uint32x2_t vmul_lane_u32(uint32x2_t a,uint32x2_t v,const int lane)
// float32x2_t vmul_lane_f32(float32x2_t a,float32x2_t v,const int lane)
// float64x1_t vmul_lane_f64(float64x1_t a,float64x1_t v,const int lane)
// float32_t vmuls_lane_f32(float32_t a,float32x2_t v,const int lane)
// float64_t vmuld_lane_f64(float64_t a,float64x1_t v,const int lane)
//
// int16x8_t vmulq_lane_s16(int16x8_t a,int16x4_t v,const int lane)
// int32x4_t vmulq_lane_s32(int32x4_t a,int32x2_t v,const int lane)
// uint16x8_t vmulq_lane_u16(uint16x8_t a,uint16x4_t v,const int lane)
// uint32x4_t vmulq_lane_u32(uint32x4_t a,uint32x2_t v,const int lane)
// float32x4_t vmulq_lane_f32(float32x4_t a,float32x2_t v,const int lane)
// float64x2_t vmulq_lane_f64(float64x2_t a,float64x1_t v,const int lane)
//
// int16x4_t vmul_laneq_s16(int16x4_t a,int16x8_t v,const int lane)
// int32x2_t vmul_laneq_s32(int32x2_t a,int32x4_t v,const int lane)
// uint16x4_t vmul_laneq_u16(uint16x4_t a,uint16x8_t v,const int lane)
// uint32x2_t vmul_laneq_u32(uint32x2_t a,uint32x4_t v,const int lane)
// float32x2_t vmul_laneq_f32(float32x2_t a,float32x4_t v,const int lane)
// float64x1_t vmul_laneq_f64(float64x1_t a,float64x2_t v,const int lane)
//
// int16x8_t vmulq_laneq_s16(int16x8_t a,int16x8_t v,const int lane)
// int32x4_t vmulq_laneq_s32(int32x4_t a,int32x4_t v,const int lane)
// uint16x8_t vmulq_laneq_u16(uint16x8_t a,uint16x8_t v,const int lane)
// uint32x4_t vmulq_laneq_u32(uint32x4_t a,uint32x4_t v,const int lane)
// float32x4_t vmulq_laneq_f32(float32x4_t a,float32x4_t v,const int lane)
// float64x2_t vmulq_laneq_f64(float64x2_t a,float64x2_t v,const int lane)
// ------------------------------------------------------------------------
// float32_t vmuls_laneq_f32(float32_t a,float32x4_t v,const int lane)
// float64_t vmuld_laneq_f64(float64_t a,float64x2_t v,const int lane)
