// 2023-04-18 16:56
#include <neon.h>
#include <neon_test.h>
/* NOTE: cge 无法 bitwise compare */
// uint8x8_t vcge_s8(int8x8_t a,int8x8_t b)
// uint16x4_t vcge_s16(int16x4_t a,int16x4_t b)
// uint32x2_t vcge_s32(int32x2_t a,int32x2_t b)
// uint64x1_t vcge_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vcge_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vcge_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vcge_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vcge_u64(uint64x1_t a,uint64x1_t b)
// uint32x2_t vcge_f32(float32x2_t a,float32x2_t b)
// uint64x1_t vcge_f64(float64x1_t a,float64x1_t b)
//
// uint8x16_t vcgeq_s8(int8x16_t a,int8x16_t b)
// uint16x8_t vcgeq_s16(int16x8_t a,int16x8_t b)
// uint32x4_t vcgeq_s32(int32x4_t a,int32x4_t b)
// uint64x2_t vcgeq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vcgeq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vcgeq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vcgeq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vcgeq_u64(uint64x2_t a,uint64x2_t b)
// uint32x4_t vcgeq_f32(float32x4_t a,float32x4_t b)
// uint64x2_t vcgeq_f64(float64x2_t a,float64x2_t b)
// -------------------------------------------------
// uint64_t vcged_s64(int64_t a,int64_t b)
//              ^---scalar
// uint64_t vcged_u64(uint64_t a,uint64_t b)
// uint32_t vcges_f32(float32_t a,float32_t b)
// uint64_t vcged_f64(float64_t a,float64_t b)
// -------------------------------------------
// uint8x8_t vcgez_s8(int8x8_t a)
//               ^---zero
// uint16x4_t vcgez_s16(int16x4_t a)
// uint32x2_t vcgez_s32(int32x2_t a)
// uint64x1_t vcgez_s64(int64x1_t a)
// uint32x2_t vcgez_f32(float32x2_t a)
// uint64x1_t vcgez_f64(float64x1_t a)
//
// uint8x16_t vcgezq_s8(int8x16_t a)
// uint16x8_t vcgezq_s16(int16x8_t a)
// uint32x4_t vcgezq_s32(int32x4_t a)
// uint64x2_t vcgezq_s64(int64x2_t a)
// uint32x4_t vcgezq_f32(float32x4_t a)
// uint64x2_t vcgezq_f64(float64x2_t a)
// -------------------------------------
// uint64_t vcgezd_s64(int64_t a)
//              ^^---zero scalar
// uint32_t vcgezs_f32(float32_t a)
// uint64_t vcgezd_f64(float64_t a)

TEST_CASE(test_vcgez_s8) {
    static const struct {
        int8_t a[8];
        uint8_t r[8];
    } test_vec[] = {
        {{0, 0, 0, 0, -64, 0, -105, -79},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, 0, 0}},
        {{55, 117, -108, -70, 111, 3, 0, -7},
         {UINT8_MAX, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0}},
        {{98, -116, 0, 34, 115, 0, -45, 47},
         {UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0,
          UINT8_MAX}},
        {{-28, 5, 22, 84, 0, -92, 0, 77},
         {0, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX,
          UINT8_MAX}},
        {{0, -123, -85, -5, 15, 126, 42, -63},
         {UINT8_MAX, 0, 0, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0}},
        {{118, 0, 0, INT8_MAX, 115, 3, -52, 0},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0,
          UINT8_MAX}},
        {{-21, 0, 89, 0, -91, -125, 0, 0},
         {0, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX}},
        {{0, 41, 0, 0, 45, -17, 0, 0},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX,
          UINT8_MAX}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        uint8x8_t r = vcgez_s8(a);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
