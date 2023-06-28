// 2023-04-21 16:41
#include <neon.h>
#include <neon_test.h>
// clang-format off
// int8x8_t vld1_s8(int8_t const *ptr)
// int16x4_t vld1_s16(int16_t const *ptr)
// int32x2_t vld1_s32(int32_t const *ptr)
// int64x1_t vld1_s64(int64_t const *ptr)
// uint8x8_t vld1_u8(uint8_t const *ptr)
// uint16x4_t vld1_u16(uint16_t const *ptr)
// uint32x2_t vld1_u32(uint32_t const *ptr)
// uint64x1_t vld1_u64(uint64_t const *ptr)
// poly64x1_t vld1_p64(poly64_t const *ptr)
// float16x4_t vld1_f16(float16_t const *ptr)
// float32x2_t vld1_f32(float32_t const *ptr)
// poly8x8_t vld1_p8(poly8_t const *ptr)
// poly16x4_t vld1_p16(poly16_t const *ptr)
// float64x1_t vld1_f64(float64_t const *ptr)
//
// int8x16_t vld1q_s8(int8_t const *ptr)
// int16x8_t vld1q_s16(int16_t const *ptr)
// int32x4_t vld1q_s32(int32_t const *ptr)
// int64x2_t vld1q_s64(int64_t const *ptr)
// uint8x16_t vld1q_u8(uint8_t const *ptr)
// uint16x8_t vld1q_u16(uint16_t const *ptr)
// uint32x4_t vld1q_u32(uint32_t const *ptr)
// uint64x2_t vld1q_u64(uint64_t const *ptr)
// poly64x2_t vld1q_p64(poly64_t const *ptr)
// float16x8_t vld1q_f16(float16_t const *ptr)
// float32x4_t vld1q_f32(float32_t const *ptr)
// poly8x16_t vld1q_p8(poly8_t const *ptr)
// poly16x8_t vld1q_p16(poly16_t const *ptr)
// float64x2_t vld1q_f64(float64_t const *ptr)
// -----------------------------------------------------------------------
// int8x8_t vld1_lane_s8(int8_t const *ptr,int8x8_t src,const int lane)
// int16x4_t vld1_lane_s16(int16_t const *ptr,int16x4_t src,const int lane)
// int32x2_t vld1_lane_s32(int32_t const *ptr,int32x2_t src,const int lane)
// int64x1_t vld1_lane_s64(int64_t const *ptr,int64x1_t src,const int lane)
// uint8x8_t vld1_lane_u8(uint8_t const *ptr,uint8x8_t src,const int lane)
// uint16x4_t vld1_lane_u16(uint16_t const *ptr,uint16x4_t src,const int lane)
// uint32x2_t vld1_lane_u32(uint32_t const *ptr,uint32x2_t src,const int lane)
// uint64x1_t vld1_lane_u64(uint64_t const *ptr,uint64x1_t src,const int lane)
// poly64x1_t vld1_lane_p64(poly64_t const *ptr,poly64x1_t src,const int lane)
// float16x4_t vld1_lane_f16(float16_t const *ptr,float16x4_t src,const int lane)
// float32x2_t vld1_lane_f32(float32_t const *ptr,float32x2_t src,const int lane)
// poly8x8_t vld1_lane_p8(poly8_t const *ptr,poly8x8_t src,const int lane)
// poly16x4_t vld1_lane_p16(poly16_t const *ptr,poly16x4_t src,const int lane)
// float64x1_t vld1_lane_f64(float64_t const *ptr,float64x1_t src,const int lane)
//
// int8x16_t vld1q_lane_s8(int8_t const *ptr,int8x16_t src,const int lane)
// int16x8_t vld1q_lane_s16(int16_t const *ptr,int16x8_t src,const int lane)
// int32x4_t vld1q_lane_s32(int32_t const *ptr,int32x4_t src,const int lane)
// int64x2_t vld1q_lane_s64(int64_t const *ptr,int64x2_t src,const int lane)
// uint8x16_t vld1q_lane_u8(uint8_t const *ptr,uint8x16_t src,const int lane)
// uint16x8_t vld1q_lane_u16(uint16_t const *ptr,uint16x8_t src,const int lane)
// uint32x4_t vld1q_lane_u32(uint32_t const *ptr,uint32x4_t src,const int lane)
// uint64x2_t vld1q_lane_u64(uint64_t const *ptr,uint64x2_t src,const int lane)
// poly64x2_t vld1q_lane_p64(poly64_t const *ptr,poly64x2_t src,const int lane)
// float16x8_t vld1q_lane_f16(float16_t const *ptr,float16x8_t src,const int lane)
// float32x4_t vld1q_lane_f32(float32_t const *ptr,float32x4_t src,const int lane)
// poly8x16_t vld1q_lane_p8(poly8_t const *ptr,poly8x16_t src,const int lane)
// poly16x8_t vld1q_lane_p16(poly16_t const *ptr,poly16x8_t src,const int lane)
// float64x2_t vld1q_lane_f64(float64_t const *ptr,float64x2_t src,const int lane)
// --------------------------------------------------------------------------------
// int8x8_t vld1_dup_s8(int8_t const *ptr)
// int16x4_t vld1_dup_s16(int16_t const *ptr)
// int32x2_t vld1_dup_s32(int32_t const *ptr)
// int64x1_t vld1_dup_s64(int64_t const *ptr)
// uint8x8_t vld1_dup_u8(uint8_t const *ptr)
// uint16x4_t vld1_dup_u16(uint16_t const *ptr)
// uint32x2_t vld1_dup_u32(uint32_t const *ptr)
// uint64x1_t vld1_dup_u64(uint64_t const *ptr)
// poly64x1_t vld1_dup_p64(poly64_t const *ptr)
// float16x4_t vld1_dup_f16(float16_t const *ptr)
// float32x2_t vld1_dup_f32(float32_t const *ptr)
// poly8x8_t vld1_dup_p8(poly8_t const *ptr)
// poly16x4_t vld1_dup_p16(poly16_t const *ptr)
// float64x1_t vld1_dup_f64(float64_t const *ptr)
//
// int8x16_t vld1q_dup_s8(int8_t const *ptr)
// int16x8_t vld1q_dup_s16(int16_t const *ptr)
// int32x4_t vld1q_dup_s32(int32_t const *ptr)
// int64x2_t vld1q_dup_s64(int64_t const *ptr)
// uint8x16_t vld1q_dup_u8(uint8_t const *ptr)
// uint16x8_t vld1q_dup_u16(uint16_t const *ptr)
// uint32x4_t vld1q_dup_u32(uint32_t const *ptr)
// uint64x2_t vld1q_dup_u64(uint64_t const *ptr)
// poly64x2_t vld1q_dup_p64(poly64_t const *ptr)
// float16x8_t vld1q_dup_f16(float16_t const *ptr)
// float32x4_t vld1q_dup_f32(float32_t const *ptr)
// poly8x16_t vld1q_dup_p8(poly8_t const *ptr)
// poly16x8_t vld1q_dup_p16(poly16_t const *ptr)
// float64x2_t vld1q_dup_f64(float64_t const *ptr)
// -------------------------------------------------
// int8x8x2_t vld2_s8(int8_t const *ptr)
// int16x4x2_t vld2_s16(int16_t const *ptr)
// int32x2x2_t vld2_s32(int32_t const *ptr)
// uint8x8x2_t vld2_u8(uint8_t const *ptr)
// uint16x4x2_t vld2_u16(uint16_t const *ptr)
// uint32x2x2_t vld2_u32(uint32_t const *ptr)
// float16x4x2_t vld2_f16(float16_t const *ptr)
// float32x2x2_t vld2_f32(float32_t const *ptr)
// poly8x8x2_t vld2_p8(poly8_t const *ptr)
// poly16x4x2_t vld2_p16(poly16_t const *ptr)
// int64x1x2_t vld2_s64(int64_t const *ptr)
// uint64x1x2_t vld2_u64(uint64_t const *ptr)
// poly64x1x2_t vld2_p64(poly64_t const *ptr)
// float64x1x2_t vld2_f64(float64_t const *ptr)
//
// int8x16x2_t vld2q_s8(int8_t const *ptr)
// int16x8x2_t vld2q_s16(int16_t const *ptr)
// int32x4x2_t vld2q_s32(int32_t const *ptr)
// uint8x16x2_t vld2q_u8(uint8_t const *ptr)
// uint16x8x2_t vld2q_u16(uint16_t const *ptr)
// uint32x4x2_t vld2q_u32(uint32_t const *ptr)
// float16x8x2_t vld2q_f16(float16_t const *ptr)
// float32x4x2_t vld2q_f32(float32_t const *ptr)
// poly8x16x2_t vld2q_p8(poly8_t const *ptr)
// poly16x8x2_t vld2q_p16(poly16_t const *ptr)
// int64x2x2_t vld2q_s64(int64_t const *ptr)
// uint64x2x2_t vld2q_u64(uint64_t const *ptr)
// poly64x2x2_t vld2q_p64(poly64_t const *ptr)
// float64x2x2_t vld2q_f64(float64_t const *ptr)
// ----------------------------------------------
// int8x8x3_t vld3_s8(int8_t const *ptr)
// int16x4x3_t vld3_s16(int16_t const *ptr)
// int32x2x3_t vld3_s32(int32_t const *ptr)
// uint8x8x3_t vld3_u8(uint8_t const *ptr)
// uint16x4x3_t vld3_u16(uint16_t const *ptr)
// uint32x2x3_t vld3_u32(uint32_t const *ptr)
// float16x4x3_t vld3_f16(float16_t const *ptr)
// float32x2x3_t vld3_f32(float32_t const *ptr)
// poly8x8x3_t vld3_p8(poly8_t const *ptr)
// poly16x4x3_t vld3_p16(poly16_t const *ptr)
// int64x1x3_t vld3_s64(int64_t const *ptr)
// uint64x1x3_t vld3_u64(uint64_t const *ptr)
// poly64x1x3_t vld3_p64(poly64_t const *ptr)
// float64x1x3_t vld3_f64(float64_t const *ptr)
//
// int8x16x3_t vld3q_s8(int8_t const *ptr)
// int16x8x3_t vld3q_s16(int16_t const *ptr)
// int32x4x3_t vld3q_s32(int32_t const *ptr)
// uint8x16x3_t vld3q_u8(uint8_t const *ptr)
// uint16x8x3_t vld3q_u16(uint16_t const *ptr)
// uint32x4x3_t vld3q_u32(uint32_t const *ptr)
// float16x8x3_t vld3q_f16(float16_t const *ptr)
// float32x4x3_t vld3q_f32(float32_t const *ptr)
// poly8x16x3_t vld3q_p8(poly8_t const *ptr)
// poly16x8x3_t vld3q_p16(poly16_t const *ptr)
// int64x2x3_t vld3q_s64(int64_t const *ptr)
// uint64x2x3_t vld3q_u64(uint64_t const *ptr)
// poly64x2x3_t vld3q_p64(poly64_t const *ptr)
// float64x2x3_t vld3q_f64(float64_t const *ptr)
// ----------------------------------------------
// int8x8x4_t vld4_s8(int8_t const *ptr)
// int16x4x4_t vld4_s16(int16_t const *ptr)
// int32x2x4_t vld4_s32(int32_t const *ptr)
// uint8x8x4_t vld4_u8(uint8_t const *ptr)
// uint16x4x4_t vld4_u16(uint16_t const *ptr)
// uint32x2x4_t vld4_u32(uint32_t const *ptr)
// float16x4x4_t vld4_f16(float16_t const *ptr)
// float32x2x4_t vld4_f32(float32_t const *ptr)
// poly8x8x4_t vld4_p8(poly8_t const *ptr)
// poly16x4x4_t vld4_p16(poly16_t const *ptr)
// int64x1x4_t vld4_s64(int64_t const *ptr)
// uint64x1x4_t vld4_u64(uint64_t const *ptr)
// poly64x1x4_t vld4_p64(poly64_t const *ptr)
// float64x1x4_t vld4_f64(float64_t const *ptr)
//
// int8x16x4_t vld4q_s8(int8_t const *ptr)
// int16x8x4_t vld4q_s16(int16_t const *ptr)
// int32x4x4_t vld4q_s32(int32_t const *ptr)
// uint8x16x4_t vld4q_u8(uint8_t const *ptr)
// uint16x8x4_t vld4q_u16(uint16_t const *ptr)
// uint32x4x4_t vld4q_u32(uint32_t const *ptr)
// float16x8x4_t vld4q_f16(float16_t const *ptr)
// float32x4x4_t vld4q_f32(float32_t const *ptr)
// poly8x16x4_t vld4q_p8(poly8_t const *ptr)
// poly16x8x4_t vld4q_p16(poly16_t const *ptr)
// int64x2x4_t vld4q_s64(int64_t const *ptr)
// uint64x2x4_t vld4q_u64(uint64_t const *ptr)
// poly64x2x4_t vld4q_p64(poly64_t const *ptr)
// float64x2x4_t vld4q_f64(float64_t const *ptr)
// ----------------------------------------------
// int8x16x2_t vld2q_dup_s8(int8_t const *ptr)
// int16x8x2_t vld2q_dup_s16(int16_t const *ptr)
// int32x4x2_t vld2q_dup_s32(int32_t const *ptr)
// uint8x16x2_t vld2q_dup_u8(uint8_t const *ptr)
// uint16x8x2_t vld2q_dup_u16(uint16_t const *ptr)
// uint32x4x2_t vld2q_dup_u32(uint32_t const *ptr)
// float16x8x2_t vld2q_dup_f16(float16_t const *ptr)
// float32x4x2_t vld2q_dup_f32(float32_t const *ptr)
// poly8x16x2_t vld2q_dup_p8(poly8_t const *ptr)
// poly16x8x2_t vld2q_dup_p16(poly16_t const *ptr)
// uint64x1x2_t vld2_dup_u64(uint64_t const *ptr)
// int64x2x2_t vld2q_dup_s64(int64_t const *ptr)
// poly64x2x2_t vld2q_dup_p64(poly64_t const *ptr)
// float64x2x2_t vld2q_dup_f64(float64_t const *ptr)
// --------------------------------------------------
// int8x16x3_t vld3q_dup_s8(int8_t const *ptr)
// int16x4x3_t vld3_dup_s16(int16_t const *ptr)
// int16x8x3_t vld3q_dup_s16(int16_t const *ptr)
// int32x2x3_t vld3_dup_s32(int32_t const *ptr)
// int32x4x3_t vld3q_dup_s32(int32_t const *ptr)
// uint8x8x3_t vld3_dup_u8(uint8_t const *ptr)
// uint8x16x3_t vld3q_dup_u8(uint8_t const *ptr)
// uint16x4x3_t vld3_dup_u16(uint16_t const *ptr)
// uint16x8x3_t vld3q_dup_u16(uint16_t const *ptr)
// uint32x2x3_t vld3_dup_u32(uint32_t const *ptr)
// uint32x4x3_t vld3q_dup_u32(uint32_t const *ptr)
// float16x4x3_t vld3_dup_f16(float16_t const *ptr)
// float16x8x3_t vld3q_dup_f16(float16_t const *ptr)
// float32x2x3_t vld3_dup_f32(float32_t const *ptr)
// float32x4x3_t vld3q_dup_f32(float32_t const *ptr)
// poly8x8x3_t vld3_dup_p8(poly8_t const *ptr)
// poly8x16x3_t vld3q_dup_p8(poly8_t const *ptr)
// poly16x4x3_t vld3_dup_p16(poly16_t const *ptr)
// poly16x8x3_t vld3q_dup_p16(poly16_t const *ptr)
// int64x1x3_t vld3_dup_s64(int64_t const *ptr)
// uint64x1x3_t vld3_dup_u64(uint64_t const *ptr)
// poly64x1x3_t vld3_dup_p64(poly64_t const *ptr)
// int64x2x3_t vld3q_dup_s64(int64_t const *ptr)
// uint64x2x3_t vld3q_dup_u64(uint64_t const *ptr)
// poly64x2x3_t vld3q_dup_p64(poly64_t const *ptr)
// float64x1x3_t vld3_dup_f64(float64_t const *ptr)
// float64x2x3_t vld3q_dup_f64(float64_t const *ptr)
// ---------------------------------------------------
// int8x8x4_t vld4_dup_s8(int8_t const *ptr)
// int8x16x4_t vld4q_dup_s8(int8_t const *ptr)
// int16x4x4_t vld4_dup_s16(int16_t const *ptr)
// int16x8x4_t vld4q_dup_s16(int16_t const *ptr)
// int32x2x4_t vld4_dup_s32(int32_t const *ptr)
// int32x4x4_t vld4q_dup_s32(int32_t const *ptr)
// uint8x8x4_t vld4_dup_u8(uint8_t const *ptr)
// uint8x16x4_t vld4q_dup_u8(uint8_t const *ptr)
// uint16x4x4_t vld4_dup_u16(uint16_t const *ptr)
// uint16x8x4_t vld4q_dup_u16(uint16_t const *ptr)
// uint32x2x4_t vld4_dup_u32(uint32_t const *ptr)
// uint32x4x4_t vld4q_dup_u32(uint32_t const *ptr)
// float16x4x4_t vld4_dup_f16(float16_t const *ptr)
// float16x8x4_t vld4q_dup_f16(float16_t const *ptr)
// float32x2x4_t vld4_dup_f32(float32_t const *ptr)
// float32x4x4_t vld4q_dup_f32(float32_t const *ptr)
// poly8x8x4_t vld4_dup_p8(poly8_t const *ptr)
// poly8x16x4_t vld4q_dup_p8(poly8_t const *ptr)
// poly16x4x4_t vld4_dup_p16(poly16_t const *ptr)
// poly16x8x4_t vld4q_dup_p16(poly16_t const *ptr)
// int64x1x4_t vld4_dup_s64(int64_t const *ptr)
// uint64x1x4_t vld4_dup_u64(uint64_t const *ptr)
// poly64x1x4_t vld4_dup_p64(poly64_t const *ptr)
// int64x2x4_t vld4q_dup_s64(int64_t const *ptr)
// uint64x2x4_t vld4q_dup_u64(uint64_t const *ptr)
// poly64x2x4_t vld4q_dup_p64(poly64_t const *ptr)
// float64x1x4_t vld4_dup_f64(float64_t const *ptr)
// float64x2x4_t vld4q_dup_f64(float64_t const *ptr)
// ---------------------------------------------------
// int16x4x2_t vld2_lane_s16(int16_t const *ptr,int16x4x2_t src,const int lane)
// int16x8x2_t vld2q_lane_s16(int16_t const *ptr,int16x8x2_t src,const int lane)
// int32x2x2_t vld2_lane_s32(int32_t const *ptr,int32x2x2_t src,const int lane)
// int32x4x2_t vld2q_lane_s32(int32_t const *ptr,int32x4x2_t src,const int lane)
// uint16x4x2_t vld2_lane_u16(uint16_t const *ptr,uint16x4x2_t src,const int lane)
// uint16x8x2_t vld2q_lane_u16(uint16_t const *ptr,uint16x8x2_t src,const int lane)
// uint32x2x2_t vld2_lane_u32(uint32_t const *ptr,uint32x2x2_t src,const int lane)
// uint32x4x2_t vld2q_lane_u32(uint32_t const *ptr,uint32x4x2_t src,const int lane)
// float16x4x2_t vld2_lane_f16(float16_t const *ptr,float16x4x2_t src,const int lane)
// float16x8x2_t vld2q_lane_f16(float16_t const *ptr,float16x8x2_t src,const int lane)
// float32x2x2_t vld2_lane_f32(float32_t const *ptr,float32x2x2_t src,const int lane)
// float32x4x2_t vld2q_lane_f32(float32_t const *ptr,float32x4x2_t src,const int lane)
// poly16x4x2_t vld2_lane_p16(poly16_t const *ptr,poly16x4x2_t src,const int lane)
// poly16x8x2_t vld2q_lane_p16(poly16_t const *ptr,poly16x8x2_t src,const int lane)
// int8x8x2_t vld2_lane_s8(int8_t const *ptr,int8x8x2_t src,const int lane)
// uint8x8x2_t vld2_lane_u8(uint8_t const *ptr,uint8x8x2_t src,const int lane)
// poly8x8x2_t vld2_lane_p8(poly8_t const *ptr,poly8x8x2_t src,const int lane)
// int8x16x2_t vld2q_lane_s8(int8_t const *ptr,int8x16x2_t src,const int lane)
// uint8x16x2_t vld2q_lane_u8(uint8_t const *ptr,uint8x16x2_t src,const int lane)
// poly8x16x2_t vld2q_lane_p8(poly8_t const *ptr,poly8x16x2_t src,const int lane)
// int64x1x2_t vld2_lane_s64(int64_t const *ptr,int64x1x2_t src,const int lane)
// int64x2x2_t vld2q_lane_s64(int64_t const *ptr,int64x2x2_t src,const int lane)
// uint64x1x2_t vld2_lane_u64(uint64_t const *ptr,uint64x1x2_t src,const int lane)
// uint64x2x2_t vld2q_lane_u64(uint64_t const *ptr,uint64x2x2_t src,const int lane)
// poly64x1x2_t vld2_lane_p64(poly64_t const *ptr,poly64x1x2_t src,const int lane)
// poly64x2x2_t vld2q_lane_p64(poly64_t const *ptr,poly64x2x2_t src,const int lane)
// float64x1x2_t vld2_lane_f64(float64_t const *ptr,float64x1x2_t src,const int lane)
// float64x2x2_t vld2q_lane_f64(float64_t const *ptr,float64x2x2_t src,const int lane)
// --------------------------------------------------------------------------------------
// int16x4x3_t vld3_lane_s16(int16_t const *ptr,int16x4x3_t src,const int lane)
// int16x8x3_t vld3q_lane_s16(int16_t const *ptr,int16x8x3_t src,const int lane)
// int32x2x3_t vld3_lane_s32(int32_t const *ptr,int32x2x3_t src,const int lane)
// int32x4x3_t vld3q_lane_s32(int32_t const *ptr,int32x4x3_t src,const int lane)
// uint16x4x3_t vld3_lane_u16(uint16_t const *ptr,uint16x4x3_t src,const int lane)
// uint16x8x3_t vld3q_lane_u16(uint16_t const *ptr,uint16x8x3_t src,const int lane)
// uint32x2x3_t vld3_lane_u32(uint32_t const *ptr,uint32x2x3_t src,const int lane)
// uint32x4x3_t vld3q_lane_u32(uint32_t const *ptr,uint32x4x3_t src,const int lane)
// float16x4x3_t vld3_lane_f16(float16_t const *ptr,float16x4x3_t src,const int lane)
// float16x8x3_t vld3q_lane_f16(float16_t const *ptr,float16x8x3_t src,const int lane)
// float32x2x3_t vld3_lane_f32(float32_t const *ptr,float32x2x3_t src,const int lane)
// float32x4x3_t vld3q_lane_f32(float32_t const *ptr,float32x4x3_t src,const int lane)
// poly16x4x3_t vld3_lane_p16(poly16_t const *ptr,poly16x4x3_t src,const int lane)
// poly16x8x3_t vld3q_lane_p16(poly16_t const *ptr,poly16x8x3_t src,const int lane)
// int8x8x3_t vld3_lane_s8(int8_t const *ptr,int8x8x3_t src,const int lane)
// uint8x8x3_t vld3_lane_u8(uint8_t const *ptr,uint8x8x3_t src,const int lane)
// poly8x8x3_t vld3_lane_p8(poly8_t const *ptr,poly8x8x3_t src,const int lane)
// int8x16x3_t vld3q_lane_s8(int8_t const *ptr,int8x16x3_t src,const int lane)
// uint8x16x3_t vld3q_lane_u8(uint8_t const *ptr,uint8x16x3_t src,const int lane)
// poly8x16x3_t vld3q_lane_p8(poly8_t const *ptr,poly8x16x3_t src,const int lane)
// int64x1x3_t vld3_lane_s64(int64_t const *ptr,int64x1x3_t src,const int lane)
// int64x2x3_t vld3q_lane_s64(int64_t const *ptr,int64x2x3_t src,const int lane)
// uint64x1x3_t vld3_lane_u64(uint64_t const *ptr,uint64x1x3_t src,const int lane)
// uint64x2x3_t vld3q_lane_u64(uint64_t const *ptr,uint64x2x3_t src,const int lane)
// poly64x1x3_t vld3_lane_p64(poly64_t const *ptr,poly64x1x3_t src,const int lane)
// poly64x2x3_t vld3q_lane_p64(poly64_t const *ptr,poly64x2x3_t src,const int lane)
// float64x1x3_t vld3_lane_f64(float64_t const *ptr,float64x1x3_t src,const int lane)
// float64x2x3_t vld3q_lane_f64(float64_t const *ptr,float64x2x3_t src,const int lane)
// -------------------------------------------------------------------------------------
// int16x4x4_t vld4_lane_s16(int16_t const *ptr,int16x4x4_t src,const int lane)
// int16x8x4_t vld4q_lane_s16(int16_t const *ptr,int16x8x4_t src,const int lane)
// int32x2x4_t vld4_lane_s32(int32_t const *ptr,int32x2x4_t src,const int lane)
// int32x4x4_t vld4q_lane_s32(int32_t const *ptr,int32x4x4_t src,const int lane)
// uint16x4x4_t vld4_lane_u16(uint16_t const *ptr,uint16x4x4_t src,const int lane)
// uint16x8x4_t vld4q_lane_u16(uint16_t const *ptr,uint16x8x4_t src,const int lane)
// uint32x2x4_t vld4_lane_u32(uint32_t const *ptr,uint32x2x4_t src,const int lane)
// uint32x4x4_t vld4q_lane_u32(uint32_t const *ptr,uint32x4x4_t src,const int lane)
// float16x4x4_t vld4_lane_f16(float16_t const *ptr,float16x4x4_t src,const int lane)
// float16x8x4_t vld4q_lane_f16(float16_t const *ptr,float16x8x4_t src,const int lane)
// float32x2x4_t vld4_lane_f32(float32_t const *ptr,float32x2x4_t src,const int lane)
// float32x4x4_t vld4q_lane_f32(float32_t const *ptr,float32x4x4_t src,const int lane)
// poly16x4x4_t vld4_lane_p16(poly16_t const *ptr,poly16x4x4_t src,const int lane)
// poly16x8x4_t vld4q_lane_p16(poly16_t const *ptr,poly16x8x4_t src,const int lane)
// int8x8x4_t vld4_lane_s8(int8_t const *ptr,int8x8x4_t src,const int lane)
// uint8x8x4_t vld4_lane_u8(uint8_t const *ptr,uint8x8x4_t src,const int lane)
// poly8x8x4_t vld4_lane_p8(poly8_t const *ptr,poly8x8x4_t src,const int lane)
// int8x16x4_t vld4q_lane_s8(int8_t const *ptr,int8x16x4_t src,const int lane)
// uint8x16x4_t vld4q_lane_u8(uint8_t const *ptr,uint8x16x4_t src,const int lane)
// poly8x16x4_t vld4q_lane_p8(poly8_t const *ptr,poly8x16x4_t src,const int lane)
// int64x1x4_t vld4_lane_s64(int64_t const *ptr,int64x1x4_t src,const int lane)
// int64x2x4_t vld4q_lane_s64(int64_t const *ptr,int64x2x4_t src,const int lane)
// uint64x1x4_t vld4_lane_u64(uint64_t const *ptr,uint64x1x4_t src,const int lane)
// uint64x2x4_t vld4q_lane_u64(uint64_t const *ptr,uint64x2x4_t src,const int lane)
// poly64x1x4_t vld4_lane_p64(poly64_t const *ptr,poly64x1x4_t src,const int lane)
// poly64x2x4_t vld4q_lane_p64(poly64_t const *ptr,poly64x2x4_t src,const int lane)
// float64x1x4_t vld4_lane_f64(float64_t const *ptr,float64x1x4_t src,const int lane)
// float64x2x4_t vld4q_lane_f64(float64_t const *ptr,float64x2x4_t src,const int lane)
// -------------------------------------------------------------------------------------
// int8x8x2_t vld1_s8_x2(int8_t const *ptr)
// int8x16x2_t vld1q_s8_x2(int8_t const *ptr)
// int16x4x2_t vld1_s16_x2(int16_t const *ptr)
// int16x8x2_t vld1q_s16_x2(int16_t const *ptr)
// int32x2x2_t vld1_s32_x2(int32_t const *ptr)
// int32x4x2_t vld1q_s32_x2(int32_t const *ptr)
// uint8x8x2_t vld1_u8_x2(uint8_t const *ptr)
// uint8x16x2_t vld1q_u8_x2(uint8_t const *ptr)
// uint16x4x2_t vld1_u16_x2(uint16_t const *ptr)
// uint16x8x2_t vld1q_u16_x2(uint16_t const *ptr)
// uint32x2x2_t vld1_u32_x2(uint32_t const *ptr)
// uint32x4x2_t vld1q_u32_x2(uint32_t const *ptr)
// float16x4x2_t vld1_f16_x2(float16_t const *ptr)
// float16x8x2_t vld1q_f16_x2(float16_t const *ptr)
// float32x2x2_t vld1_f32_x2(float32_t const *ptr)
// float32x4x2_t vld1q_f32_x2(float32_t const *ptr)
// poly8x8x2_t vld1_p8_x2(poly8_t const *ptr)
// poly8x16x2_t vld1q_p8_x2(poly8_t const *ptr)
// poly16x4x2_t vld1_p16_x2(poly16_t const *ptr)
// poly16x8x2_t vld1q_p16_x2(poly16_t const *ptr)
// int64x1x2_t vld1_s64_x2(int64_t const *ptr)
// uint64x1x2_t vld1_u64_x2(uint64_t const *ptr)
// poly64x1x2_t vld1_p64_x2(poly64_t const *ptr)
// int64x2x2_t vld1q_s64_x2(int64_t const *ptr)
// uint64x2x2_t vld1q_u64_x2(uint64_t const *ptr)
// poly64x2x2_t vld1q_p64_x2(poly64_t const *ptr)
// float64x1x2_t vld1_f64_x2(float64_t const *ptr)
// float64x2x2_t vld1q_f64_x2(float64_t const *ptr)
// --------------------------------------------------
// int8x8x3_t vld1_s8_x3(int8_t const *ptr)
// int8x16x3_t vld1q_s8_x3(int8_t const *ptr)
// int16x4x3_t vld1_s16_x3(int16_t const *ptr)
// int16x8x3_t vld1q_s16_x3(int16_t const *ptr)
// int32x2x3_t vld1_s32_x3(int32_t const *ptr)
// int32x4x3_t vld1q_s32_x3(int32_t const *ptr)
// uint8x8x3_t vld1_u8_x3(uint8_t const *ptr)
// uint8x16x3_t vld1q_u8_x3(uint8_t const *ptr)
// uint16x4x3_t vld1_u16_x3(uint16_t const *ptr)
// uint16x8x3_t vld1q_u16_x3(uint16_t const *ptr)
// uint32x2x3_t vld1_u32_x3(uint32_t const *ptr)
// uint32x4x3_t vld1q_u32_x3(uint32_t const *ptr)
// float16x4x3_t vld1_f16_x3(float16_t const *ptr)
// float16x8x3_t vld1q_f16_x3(float16_t const *ptr)
// float32x2x3_t vld1_f32_x3(float32_t const *ptr)
// float32x4x3_t vld1q_f32_x3(float32_t const *ptr)
// poly8x8x3_t vld1_p8_x3(poly8_t const *ptr)
// poly8x16x3_t vld1q_p8_x3(poly8_t const *ptr)
// poly16x4x3_t vld1_p16_x3(poly16_t const *ptr)
// poly16x8x3_t vld1q_p16_x3(poly16_t const *ptr)
// int64x1x3_t vld1_s64_x3(int64_t const *ptr)
// uint64x1x3_t vld1_u64_x3(uint64_t const *ptr)
// poly64x1x3_t vld1_p64_x3(poly64_t const *ptr)
// int64x2x3_t vld1q_s64_x3(int64_t const *ptr)
// uint64x2x3_t vld1q_u64_x3(uint64_t const *ptr)
// poly64x2x3_t vld1q_p64_x3(poly64_t const *ptr)
// float64x1x3_t vld1_f64_x3(float64_t const *ptr)
// float64x2x3_t vld1q_f64_x3(float64_t const *ptr)
// --------------------------------------------------
// int8x8x4_t vld1_s8_x4(int8_t const *ptr)
// int8x16x4_t vld1q_s8_x4(int8_t const *ptr)
// int16x4x4_t vld1_s16_x4(int16_t const *ptr)
// int16x8x4_t vld1q_s16_x4(int16_t const *ptr)
// int32x2x4_t vld1_s32_x4(int32_t const *ptr)
// int32x4x4_t vld1q_s32_x4(int32_t const *ptr)
// uint8x8x4_t vld1_u8_x4(uint8_t const *ptr)
// uint8x16x4_t vld1q_u8_x4(uint8_t const *ptr)
// uint16x4x4_t vld1_u16_x4(uint16_t const *ptr)
// uint16x8x4_t vld1q_u16_x4(uint16_t const *ptr)
// uint32x2x4_t vld1_u32_x4(uint32_t const *ptr)
// uint32x4x4_t vld1q_u32_x4(uint32_t const *ptr)
// float16x4x4_t vld1_f16_x4(float16_t const *ptr)
// float16x8x4_t vld1q_f16_x4(float16_t const *ptr)
// float32x2x4_t vld1_f32_x4(float32_t const *ptr)
// float32x4x4_t vld1q_f32_x4(float32_t const *ptr)
// poly8x8x4_t vld1_p8_x4(poly8_t const *ptr)
// poly8x16x4_t vld1q_p8_x4(poly8_t const *ptr)
// poly16x4x4_t vld1_p16_x4(poly16_t const *ptr)
// poly16x8x4_t vld1q_p16_x4(poly16_t const *ptr)
// int64x1x4_t vld1_s64_x4(int64_t const *ptr)
// uint64x1x4_t vld1_u64_x4(uint64_t const *ptr)
// poly64x1x4_t vld1_p64_x4(poly64_t const *ptr)
// int64x2x4_t vld1q_s64_x4(int64_t const *ptr)
// uint64x2x4_t vld1q_u64_x4(uint64_t const *ptr)
// poly64x2x4_t vld1q_p64_x4(poly64_t const *ptr)
// float64x1x4_t vld1_f64_x4(float64_t const *ptr)
// float64x2x4_t vld1q_f64_x4(float64_t const *ptr)
// clang-format on

TEST_CASE(test_vld1_lane_s8) {
    static const struct {
        int8_t src[8];
        int8_t buf;
        int8_t r[8];
    } test_vec[] = {
        {{-55, -75, -8, 112, INT8_MIN, 101, 50, 24},
         45,
         {45, -75, -8, 112, INT8_MIN, 101, 50, 24}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t src = vld1_s8(test_vec[i].src);
        int8x8_t r = vld1_lane_s8(&test_vec[i].buf, src, 0);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
/*
TEST_CASE(test_vld1_dup_s8) {
    static const struct {
        int8_t a;
        int8_t r[8];
    } test_vec[] = {
        {73, {73, 73, 73, 73, 73, 73, 73, 73}},
        {-124, {-124, -124, -124, -124, -124, -124, -124, -124}},
        {26, {26, 26, 26, 26, 26, 26, 26, 26}},
        {-89, {-89, -89, -89, -89, -89, -89, -89, -89}},
        {76, {76, 76, 76, 76, 76, 76, 76, 76}},
        {-91, {-91, -91, -91, -91, -91, -91, -91, -91}},
        {29, {29, 29, 29, 29, 29, 29, 29, 29}},
        {-28, {-28, -28, -28, -28, -28, -28, -28, -28}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t r = vld1_dup_s8(&test_vec[i].a);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vld2_s8) {
    static const struct {
        int8_t a[16];
        int8_t r[2][8];
    } test_vec[] = {
        {{-68, 76, 126, -73, -89, 54, 92, -47, -92, 107, -48, 54, 35, -53, -80,
          34},
         {{-68, 126, -89, 92, -92, -48, 35, -80},
          {76, -73, 54, -47, 107, 54, -53, 34}}},
        {{-90, -85, 93, -59, -121, -55, 2, -111, 49, 61, 4, -65, 56, 54, -101,
          -12},
         {{-90, 93, -121, 2, 49, 4, 56, -101},
          {-85, -59, -55, -111, 61, -65, 54, -12}}},
        {{-126, 25, -85, 41, 80, 8, -5, -12, 115, -53, 42, -105, -106, -38, -71,
          61},
         {{-126, -85, 80, -5, 115, 42, -106, -71},
          {25, 41, 8, -12, -53, -105, -38, 61}}},
        {{-123, 22, 2, 12, -33, 4, -99, 16, 65, -94, -49, 122, -40, 107, 110,
          90},
         {{-123, 2, -33, -99, 65, -49, -40, 110},
          {22, 12, 4, 16, -94, 122, 107, 90}}},
        {{-124, 26, -125, -44, 34, 126, -56, -107, 74, -14, 44, -32, -52, -27,
          29, 82},
         {{-124, -125, 34, -56, 74, 44, -52, 29},
          {26, -44, 126, -107, -14, -32, -27, 82}}},
        {{-4, 31, 94, -37, 35, -4, -20, 101, -98, -69, -33, 118, 38, 77, -48,
          -85},
         {{-4, 94, 35, -20, -98, -33, 38, -48},
          {31, -37, -4, 101, -69, 118, 77, -85}}},
        {{103, 83, INT8_MAX, -119, -46, 72, 31, 28, 58, 75, -4, 7, 49, 26, 89,
          45},
         {{103, INT8_MAX, -46, 31, 58, -4, 49, 89},
          {83, -119, 72, 28, 75, 7, 26, 45}}},
        {{57, -73, 8, 93, -77, -12, -62, 81, -80, -95, -57, -42, -18, -105,
          -127, 86},
         {{57, 8, -77, -62, -80, -57, -18, -127},
          {-73, 93, -12, 81, -95, -42, -105, 86}}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8x2_t r = vld2_s8(test_vec[i].a);
        int8x8_t check0 = vld1_s8(test_vec[i].r[0]);
        int8x8_t check1 = vld1_s8(test_vec[i].r[1]);
        ASSERT_EQUAL(r.val[0], check0);
        ASSERT_EQUAL(r.val[1], check1);
    }
    return 0;
}
*/
