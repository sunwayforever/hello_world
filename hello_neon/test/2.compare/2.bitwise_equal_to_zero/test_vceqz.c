// 2023-04-18 16:24
#include <neon.h>
#include <neon_test.h>
// uint8x8_t vceqz_s8(int8x8_t a)
//            ^^^^---compare bitwise equal zero
// uint16x4_t vceqz_s16(int16x4_t a)
// uint32x2_t vceqz_s32(int32x2_t a)
// uint64x1_t vceqz_s64(int64x1_t a)
// uint8x8_t vceqz_u8(uint8x8_t a)
// uint16x4_t vceqz_u16(uint16x4_t a)
// uint32x2_t vceqz_u32(uint32x2_t a)
// uint64x1_t vceqz_u64(uint64x1_t a)
// uint32x2_t vceqz_f32(float32x2_t a)
// uint64x1_t vceqz_f64(float64x1_t a)
//
// uint8x16_t vceqzq_s8(int8x16_t a)
// uint16x8_t vceqzq_s16(int16x8_t a)
// uint32x4_t vceqzq_s32(int32x4_t a)
// uint64x2_t vceqzq_s64(int64x2_t a)
// uint8x16_t vceqzq_u8(uint8x16_t a)
// uint16x8_t vceqzq_u16(uint16x8_t a)
// uint32x4_t vceqzq_u32(uint32x4_t a)
// uint64x2_t vceqzq_u64(uint64x2_t a)
// uint32x4_t vceqzq_f32(float32x4_t a)
// uint64x2_t vceqzq_f64(float64x2_t a)
// ------------------------------------
// uint8x16_t vceqzq_p8(poly8x16_t a)
// uint64x1_t vceqz_p64(poly64x1_t a)
// uint64x2_t vceqzq_p64(poly64x2_t a)
// ------------------------------------
// uint64_t vceqzd_s64(int64_t a)
// uint64_t vceqzd_u64(uint64_t a)
// uint32_t vceqzs_f32(float32_t a)
// uint64_t vceqzd_f64(float64_t a)
