// 2023-04-21 14:18
#include <neon.h>
#include <neon_test.h>
// int8_t vdupb_lane_s8(int8x8_t vec,const int lane)
// int16_t vduph_lane_s16(int16x4_t vec,const int lane)
// int32_t vdups_lane_s32(int32x2_t vec,const int lane)
// int64_t vdupd_lane_s64(int64x1_t vec,const int lane)
// uint8_t vdupb_lane_u8(uint8x8_t vec,const int lane)
// uint16_t vduph_lane_u16(uint16x4_t vec,const int lane)
// uint32_t vdups_lane_u32(uint32x2_t vec,const int lane)
// uint64_t vdupd_lane_u64(uint64x1_t vec,const int lane)
// float32_t vdups_lane_f32(float32x2_t vec,const int lane)
// float64_t vdupd_lane_f64(float64x1_t vec,const int lane)
// poly8_t vdupb_lane_p8(poly8x8_t vec,const int lane)
// poly16_t vduph_lane_p16(poly16x4_t vec,const int lane)
// ----------------------------------------------------------
// int8_t vdupb_laneq_s8(int8x16_t vec,const int lane)
// int16_t vduph_laneq_s16(int16x8_t vec,const int lane)
// int32_t vdups_laneq_s32(int32x4_t vec,const int lane)
// int64_t vdupd_laneq_s64(int64x2_t vec,const int lane)
// uint8_t vdupb_laneq_u8(uint8x16_t vec,const int lane)
// uint16_t vduph_laneq_u16(uint16x8_t vec,const int lane)
// uint32_t vdups_laneq_u32(uint32x4_t vec,const int lane)
// uint64_t vdupd_laneq_u64(uint64x2_t vec,const int lane)
// float32_t vdups_laneq_f32(float32x4_t vec,const int lane)
// float64_t vdupd_laneq_f64(float64x2_t vec,const int lane)
// poly8_t vdupb_laneq_p8(poly8x16_t vec,const int lane)
// poly16_t vduph_laneq_p16(poly16x8_t vec,const int lane)
// ----------------------------------------------------------
// uint8_t vget_lane_u8(uint8x8_t v,const int lane)
// uint16_t vget_lane_u16(uint16x4_t v,const int lane)
// uint32_t vget_lane_u32(uint32x2_t v,const int lane)
// uint64_t vget_lane_u64(uint64x1_t v,const int lane)
// poly64_t vget_lane_p64(poly64x1_t v,const int lane)
// int8_t vget_lane_s8(int8x8_t v,const int lane)
// int16_t vget_lane_s16(int16x4_t v,const int lane)
// int32_t vget_lane_s32(int32x2_t v,const int lane)
// int64_t vget_lane_s64(int64x1_t v,const int lane)
// poly8_t vget_lane_p8(poly8x8_t v,const int lane)
// poly16_t vget_lane_p16(poly16x4_t v,const int lane)
// float32_t vget_lane_f32(float32x2_t v,const int lane)
// float64_t vget_lane_f64(float64x1_t v,const int lane)
//
// uint8_t vgetq_lane_u8(uint8x16_t v,const int lane)
// uint16_t vgetq_lane_u16(uint16x8_t v,const int lane)
// uint32_t vgetq_lane_u32(uint32x4_t v,const int lane)
// uint64_t vgetq_lane_u64(uint64x2_t v,const int lane)
// poly64_t vgetq_lane_p64(poly64x2_t v,const int lane)
// int8_t vgetq_lane_s8(int8x16_t v,const int lane)
// int16_t vgetq_lane_s16(int16x8_t v,const int lane)
// int32_t vgetq_lane_s32(int32x4_t v,const int lane)
// int64_t vgetq_lane_s64(int64x2_t v,const int lane)
// poly8_t vgetq_lane_p8(poly8x16_t v,const int lane)
// poly16_t vgetq_lane_p16(poly16x8_t v,const int lane)
// float16_t vget_lane_f16(float16x4_t v,const int lane)
// float16_t vgetq_lane_f16(float16x8_t v,const int lane)
// float32_t vgetq_lane_f32(float32x4_t v,const int lane)
// float64_t vgetq_lane_f64(float64x2_t v,const int lane)

TEST_CASE(test_vget_lane_s8) {
    struct {
        int8_t a[8];
        int8_t r;
    } test_vec[] = {
        {{79, 68, -36, 47, 87, -22, -44, -111}, 68},
        {{-78, 75, -106, 111, -55, 39, -69, -110}, 75},
        {{72, -120, -122, -86, 90, -24, -60, -104}, -120},
        {{116, 37, -99, -48, 117, -31, -84, -92}, 37},
        {{-106, 120, -54, -64, 42, 21, 87, -103}, 120},
        {{126, 84, 112, 98, -100, -7, -23, 70}, 84},
        {{-47, 11, -21, 10, INT8_MAX, 17, -89, 79}, 11},
        {{-120, -5, 42, -64, -110, -94, -118, 82}, -5},

    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8_t r = vget_lane_s8(a, 1);
        ASSERT_EQUAL_SCALAR(r, test_vec[i].r);
    }
    return 0;
}
