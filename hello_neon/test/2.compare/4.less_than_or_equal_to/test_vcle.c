// 2023-04-18 17:11
#include <neon.h>
#include <neon_test.h>
// uint8x8_t vcle_s8(int8x8_t a,int8x8_t b)
// uint16x4_t vcle_s16(int16x4_t a,int16x4_t b)
// uint32x2_t vcle_s32(int32x2_t a,int32x2_t b)
// uint64x1_t vcle_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vcle_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vcle_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vcle_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vcle_u64(uint64x1_t a,uint64x1_t b)
// uint32x2_t vcle_f32(float32x2_t a,float32x2_t b)
// uint64x1_t vcle_f64(float64x1_t a,float64x1_t b)
//
// uint8x16_t vcleq_s8(int8x16_t a,int8x16_t b)
// uint16x8_t vcleq_s16(int16x8_t a,int16x8_t b)
// uint32x4_t vcleq_s32(int32x4_t a,int32x4_t b)
// uint64x2_t vcleq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vcleq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vcleq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vcleq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vcleq_u64(uint64x2_t a,uint64x2_t b)
// uint32x4_t vcleq_f32(float32x4_t a,float32x4_t b)
// uint64x2_t vcleq_f64(float64x2_t a,float64x2_t b)
// -------------------------------------------------
// uint64_t vcled_s64(int64_t a,int64_t b)
// uint64_t vcled_u64(uint64_t a,uint64_t b)
// uint32_t vcles_f32(float32_t a,float32_t b)
// uint64_t vcled_f64(float64_t a,float64_t b)
// --------------------------------------------
// uint8x8_t vclez_s8(int8x8_t a)
// uint16x4_t vclez_s16(int16x4_t a)
// uint32x2_t vclez_s32(int32x2_t a)
// uint64x1_t vclez_s64(int64x1_t a)
// uint32x2_t vclez_f32(float32x2_t a)
// uint64x1_t vclez_f64(float64x1_t a)
//
// uint8x16_t vclezq_s8(int8x16_t a)
// uint16x8_t vclezq_s16(int16x8_t a)
// uint32x4_t vclezq_s32(int32x4_t a)
// uint64x2_t vclezq_s64(int64x2_t a)
// uint32x4_t vclezq_f32(float32x4_t a)
// uint64x2_t vclezq_f64(float64x2_t a)
// ------------------------------------
// uint64_t vclezd_s64(int64_t a)
// uint32_t vclezs_f32(float32_t a)
// uint64_t vclezd_f64(float64_t a)