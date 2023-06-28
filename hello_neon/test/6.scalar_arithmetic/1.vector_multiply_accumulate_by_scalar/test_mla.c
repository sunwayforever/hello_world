// 2023-04-20 16:06
#include <neon.h>
#include <neon_test.h>
// int16x4_t vmla_lane_s16(int16x4_t a,int16x4_t b,int16x4_t v,const int lane)
// int32x2_t vmla_lane_s32(int32x2_t a,int32x2_t b,int32x2_t v,const int lane)
// uint16x4_t vmla_lane_u16(uint16x4_t a,uint16x4_t b,uint16x4_t v,const int
// lane) uint32x2_t vmla_lane_u32(uint32x2_t a,uint32x2_t b,uint32x2_t v,const
// int lane) float32x2_t vmla_lane_f32(float32x2_t a,float32x2_t b,float32x2_t
// v,const int lane)
//
// int16x4_t vmla_laneq_s16(int16x4_t a,int16x4_t b,int16x8_t v,const int lane)
// int32x2_t vmla_laneq_s32(int32x2_t a,int32x2_t b,int32x4_t v,const int lane)
// uint16x4_t vmla_laneq_u16(uint16x4_t a,uint16x4_t b,uint16x8_t v,const int
// lane) uint32x2_t vmla_laneq_u32(uint32x2_t a,uint32x2_t b,uint32x4_t v,const
// int lane) float32x2_t vmla_laneq_f32(float32x2_t a,float32x2_t b,float32x4_t
// v,const int lane)
//
// int16x8_t vmlaq_lane_s16(int16x8_t a,int16x8_t b,int16x4_t v,const int lane)
// int32x4_t vmlaq_lane_s32(int32x4_t a,int32x4_t b,int32x2_t v,const int lane)
// uint16x8_t vmlaq_lane_u16(uint16x8_t a,uint16x8_t b,uint16x4_t v,const int
// lane) uint32x4_t vmlaq_lane_u32(uint32x4_t a,uint32x4_t b,uint32x2_t v,const
// int lane) float32x4_t vmlaq_lane_f32(float32x4_t a,float32x4_t b,float32x2_t
// v,const int lane)
//
// int16x8_t vmlaq_laneq_s16(int16x8_t a,int16x8_t b,int16x8_t v,const int lane)
// int32x4_t vmlaq_laneq_s32(int32x4_t a,int32x4_t b,int32x4_t v,const int lane)
// uint16x8_t vmlaq_laneq_u16(uint16x8_t a,uint16x8_t b,uint16x8_t v,const int
// lane) uint32x4_t vmlaq_laneq_u32(uint32x4_t a,uint32x4_t b,uint32x4_t v,const
// int lane)
// ---------------------------------------------------------------------------------
// widen:
// int32x4_t vmlal_lane_s16(int32x4_t a,int16x4_t b,int16x4_t v,const int lane)
// int64x2_t vmlal_lane_s32(int64x2_t a,int32x2_t b,int32x2_t v,const int lane)
// uint32x4_t vmlal_lane_u16(uint32x4_t a,uint16x4_t b,uint16x4_t v,const int
// lane) uint64x2_t vmlal_lane_u32(uint64x2_t a,uint32x2_t b,uint32x2_t v,const
// int lane) int32x4_t vmlal_high_lane_s16(int32x4_t a,int16x8_t b,int16x4_t
// v,const int lane) int64x2_t vmlal_high_lane_s32(int64x2_t a,int32x4_t
// b,int32x2_t v,const int lane) uint32x4_t vmlal_high_lane_u16(uint32x4_t
// a,uint16x8_t b,uint16x4_t v,const int lane) uint64x2_t
// vmlal_high_lane_u32(uint64x2_t a,uint32x4_t b,uint32x2_t v,const int lane)
//
// int32x4_t vmlal_laneq_s16(int32x4_t a,int16x4_t b,int16x8_t v,const int lane)
// int64x2_t vmlal_laneq_s32(int64x2_t a,int32x2_t b,int32x4_t v,const int lane)
// uint32x4_t vmlal_laneq_u16(uint32x4_t a,uint16x4_t b,uint16x8_t v,const int
// lane) uint64x2_t vmlal_laneq_u32(uint64x2_t a,uint32x2_t b,uint32x4_t v,const
// int lane) int32x4_t vmlal_high_laneq_s16(int32x4_t a,int16x8_t b,int16x8_t
// v,const int lane) int64x2_t vmlal_high_laneq_s32(int64x2_t a,int32x4_t
// b,int32x4_t v,const int lane) uint32x4_t vmlal_high_laneq_u16(uint32x4_t
// a,uint16x8_t b,uint16x8_t v,const int lane) uint64x2_t
// vmlal_high_laneq_u32(uint64x2_t a,uint32x4_t b,uint32x4_t v,const int lane)
// --------------------------------------------------------------------------------------
// scalar:
// int16x4_t vmla_n_s16(int16x4_t a,int16x4_t b,int16_t c)
// int32x2_t vmla_n_s32(int32x2_t a,int32x2_t b,int32_t c)
// uint16x4_t vmla_n_u16(uint16x4_t a,uint16x4_t b,uint16_t c)
// uint32x2_t vmla_n_u32(uint32x2_t a,uint32x2_t b,uint32_t c)
// float32x2_t vmla_n_f32(float32x2_t a,float32x2_t b,float32_t c)
//
// int16x8_t vmlaq_n_s16(int16x8_t a,int16x8_t b,int16_t c)
// int32x4_t vmlaq_n_s32(int32x4_t a,int32x4_t b,int32_t c)
// uint16x8_t vmlaq_n_u16(uint16x8_t a,uint16x8_t b,uint16_t c)
// uint32x4_t vmlaq_n_u32(uint32x4_t a,uint32x4_t b,uint32_t c)
// float32x4_t vmlaq_n_f32(float32x4_t a,float32x4_t b,float32_t c)
