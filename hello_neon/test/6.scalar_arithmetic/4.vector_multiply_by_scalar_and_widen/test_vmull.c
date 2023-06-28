// 2023-04-20 16:26
#include <neon.h>
#include <neon_test.h>
// int32x4_t vmull_n_s16(int16x4_t a,int16_t b)
// int64x2_t vmull_n_s32(int32x2_t a,int32_t b)
// uint32x4_t vmull_n_u16(uint16x4_t a,uint16_t b)
// uint64x2_t vmull_n_u32(uint32x2_t a,uint32_t b)
// -------------------------------------------------
// int32x4_t vmull_high_n_s16(int16x8_t a,int16_t b)
// int64x2_t vmull_high_n_s32(int32x4_t a,int32_t b)
// uint32x4_t vmull_high_n_u16(uint16x8_t a,uint16_t b)
// uint64x2_t vmull_high_n_u32(uint32x4_t a,uint32_t b)
// -----------------------------------------------------------------
// int32x4_t vmull_lane_s16(int16x4_t a,int16x4_t v,const int lane)
// int64x2_t vmull_lane_s32(int32x2_t a,int32x2_t v,const int lane)
// uint32x4_t vmull_lane_u16(uint16x4_t a,uint16x4_t v,const int lane)
// uint64x2_t vmull_lane_u32(uint32x2_t a,uint32x2_t v,const int lane)
//
// int32x4_t vmull_laneq_s16(int16x4_t a,int16x8_t v,const int lane)
// int64x2_t vmull_laneq_s32(int32x2_t a,int32x4_t v,const int lane)
// uint32x4_t vmull_laneq_u16(uint16x4_t a,uint16x8_t v,const int lane)
// uint64x2_t vmull_laneq_u32(uint32x2_t a,uint32x4_t v,const int lane)
// ----------------------------------------------------------------------
// int32x4_t vmull_high_lane_s16(int16x8_t a,int16x4_t v,const int lane)
// int64x2_t vmull_high_lane_s32(int32x4_t a,int32x2_t v,const int lane)
// uint32x4_t vmull_high_lane_u16(uint16x8_t a,uint16x4_t v,const int lane)
// uint64x2_t vmull_high_lane_u32(uint32x4_t a,uint32x2_t v,const int lane)
//
// int32x4_t vmull_high_laneq_s16(int16x8_t a,int16x8_t v,const int lane)
// int64x2_t vmull_high_laneq_s32(int32x4_t a,int32x4_t v,const int lane)
// uint32x4_t vmull_high_laneq_u16(uint16x8_t a,uint16x8_t v,const int lane)
// uint64x2_t vmull_high_laneq_u32(uint32x4_t a,uint32x4_t v,const int lane)
