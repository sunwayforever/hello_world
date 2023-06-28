// 2023-04-20 16:15
#include <neon.h>
#include <neon_test.h>
// int16x4_t vmls_lane_s16(int16x4_t a,int16x4_t b,int16x4_t v,const int lane)
// int32x2_t vmls_lane_s32(int32x2_t a,int32x2_t b,int32x2_t v,const int lane)
// uint16x4_t vmls_lane_u16(uint16x4_t a,uint16x4_t b,uint16x4_t v,const int lane)
// uint32x2_t vmls_lane_u32(uint32x2_t a,uint32x2_t b,uint32x2_t v,const int lane)
// float32x2_t vmls_lane_f32(float32x2_t a,float32x2_t b,float32x2_t v,const int lane)
//
// int16x4_t vmls_laneq_s16(int16x4_t a,int16x4_t b,int16x8_t v,const int lane)
// int32x2_t vmls_laneq_s32(int32x2_t a,int32x2_t b,int32x4_t v,const int lane)
// uint16x4_t vmls_laneq_u16(uint16x4_t a,uint16x4_t b,uint16x8_t v,const int lane)
// uint32x2_t vmls_laneq_u32(uint32x2_t a,uint32x2_t b,uint32x4_t v,const int lane)
// float32x2_t vmls_laneq_f32(float32x2_t a,float32x2_t b,float32x4_t v,const int lane)
//
// int16x8_t vmlsq_lane_s16(int16x8_t a,int16x8_t b,int16x4_t v,const int lane)
// int32x4_t vmlsq_lane_s32(int32x4_t a,int32x4_t b,int32x2_t v,const int lane)
// uint16x8_t vmlsq_lane_u16(uint16x8_t a,uint16x8_t b,uint16x4_t v,const int lane)
// uint32x4_t vmlsq_lane_u32(uint32x4_t a,uint32x4_t b,uint32x2_t v,const int lane)
//
// int16x8_t vmlsq_laneq_s16(int16x8_t a,int16x8_t b,int16x8_t v,const int lane)
// int32x4_t vmlsq_laneq_s32(int32x4_t a,int32x4_t b,int32x4_t v,const int lane)
// uint16x8_t vmlsq_laneq_u16(uint16x8_t a,uint16x8_t b,uint16x8_t v,const int lane)
// uint32x4_t vmlsq_laneq_u32(uint32x4_t a,uint32x4_t b,uint32x4_t v,const int lane)
// ----------------------------------------------------------------------------------
// widen:
// int32x4_t vmlsl_lane_s16(int32x4_t a,int16x4_t b,int16x4_t v,const int lane)
// int64x2_t vmlsl_lane_s32(int64x2_t a,int32x2_t b,int32x2_t v,const int lane)
// uint32x4_t vmlsl_lane_u16(uint32x4_t a,uint16x4_t b,uint16x4_t v,const int lane)
// uint64x2_t vmlsl_lane_u32(uint64x2_t a,uint32x2_t b,uint32x2_t v,const int lane)
//
// int32x4_t vmlsl_laneq_s16(int32x4_t a,int16x4_t b,int16x8_t v,const int lane)
// int64x2_t vmlsl_laneq_s32(int64x2_t a,int32x2_t b,int32x4_t v,const int lane)
// uint32x4_t vmlsl_laneq_u16(uint32x4_t a,uint16x4_t b,uint16x8_t v,const int lane)
// uint64x2_t vmlsl_laneq_u32(uint64x2_t a,uint32x2_t b,uint32x4_t v,const int lane)
// ----------------------------------------------------------------------------------
// high:
// int32x4_t vmlsl_high_lane_s16(int32x4_t a,int16x8_t b,int16x4_t v,const int lane)
// int64x2_t vmlsl_high_lane_s32(int64x2_t a,int32x4_t b,int32x2_t v,const int lane)
// uint32x4_t vmlsl_high_lane_u16(uint32x4_t a,uint16x8_t b,uint16x4_t v,const int lane)
// uint64x2_t vmlsl_high_lane_u32(uint64x2_t a,uint32x4_t b,uint32x2_t v,const int lane)
//
// int32x4_t vmlsl_high_laneq_s16(int32x4_t a,int16x8_t b,int16x8_t v,const int lane)
// int64x2_t vmlsl_high_laneq_s32(int64x2_t a,int32x4_t b,int32x4_t v,const int lane)
// uint32x4_t vmlsl_high_laneq_u16(uint32x4_t a,uint16x8_t b,uint16x8_t v,const int lane)
// uint64x2_t vmlsl_high_laneq_u32(uint64x2_t a,uint32x4_t b,uint32x4_t v,const int lane)
