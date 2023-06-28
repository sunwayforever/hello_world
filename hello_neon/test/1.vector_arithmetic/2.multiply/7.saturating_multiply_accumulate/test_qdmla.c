// 2023-04-17 13:41
#include <neon.h>
#include <neon_test.h>
/* clang-format off */
//  r[i]=a[i]+((int32_t)b[i]*c[i]*2)
//  int32x4_t vqdmlal_s16(int32x4_t a,int16x4_t b,int16x4_t c)
//             ^^---^ saturating doubling
//                  +-- widen
//  int64x2_t vqdmlal_s32(int64x2_t a,int32x2_t b,int32x2_t c)
//  ----------------------------------------------------------
//  int32_t vqdmlalh_s16(int32_t a,int16_t b,int16_t c)
//                 ^--- scalar, HI
//  int64_t vqdmlals_s32(int64_t a,int32_t b,int32_t c)
//  ----------------------------------------------------------
//  r[i]=a[i]+((int32_t)b[i+8]*c[i+8]*2)
//  int32x4_t vqd                       mlal_        high_s16(int32x4_t a,int16x8_t b,int16x8_t c)
//             ^^---saturating doubling    ^---widen ^^^^--- high
//  int64x2_t vqdmlal_high_s32(int64x2_t a,int32x4_t b,int32x4_t c)
//  ----------------------------------------------------------
//  int32x4_t vqdmlsl_s16(int32x4_t a,int16x4_t b,int16x4_t c)
//                 ^---subtract
//  int64x2_t vqdmlsl_s32(int64x2_t a,int32x2_t b,int32x2_t c)
//  int32_t vqdmlslh_s16(int32_t a,int16_t b,int16_t c)
//  int64_t vqdmlsls_s32(int64_t a,int32_t b,int32_t c)
//  int32x4_t vqdmlsl_high_s16(int32x4_t a,int16x8_t b,int16x8_t c)
//  int64x2_t vqdmlsl_high_s32(int64x2_t a,int32x4_t b,int32x4_t c)
//  ---------------------------------------------------------------
//  int32x4_t vqdmlal_lane_s16(int32x4_t a,int16x4_t b,int16x4_t v,const int lane)
//                    ^^^^--- lane, r[i]=a[i]+((int32_t)b[i]*v[lane]*2)
//  int64x2_t vqdmlal_lane_s32(int64x2_t a,int32x2_t b,int32x2_t v,const int lane)
//  int32_t vqdmlalh_lane_s16(int32_t a,int16_t b,int16x4_t v,const int lane)
//  int64_t vqdmlals_lane_s32(int64_t a,int32_t b,int32x2_t v,const int lane)
//  int32x4_t vqdmlal_high_lane_s16(int32x4_t a,int16x8_t b,int16x4_t v,const int lane)
//  int64x2_t vqdmlal_high_lane_s32(int64x2_t a,int32x4_t b,int32x2_t v,const int lane)
//  int32x4_t vqdmlal_laneq_s16(int32x4_t a,int16x4_t b,int16x8_t v,const int lane)
//  int64x2_t vqdmlal_laneq_s32(int64x2_t a,int32x2_t b,int32x4_t v,const int lane)
//  int32_t vqdmlalh_laneq_s16(int32_t a,int16_t b,int16x8_t v,const int lane)
//  int64_t vqdmlals_laneq_s32(int64_t a,int32_t b,int32x4_t v,const int lane)
//  int32x4_t vqdmlal_high_laneq_s16(int32x4_t a,int16x8_t b,int16x8_t v,const int lane)
//  int64x2_t vqdmlal_high_laneq_s32(int64x2_t a,int32x4_t b,int32x4_t v,const int lane)
//  int32x4_t vqdmlsl_lane_s16(int32x4_t a,int16x4_t b,int16x4_t v,const int lane)
//  int64x2_t vqdmlsl_lane_s32(int64x2_t a,int32x2_t b,int32x2_t v,const int lane)
//  int32_t vqdmlslh_lane_s16(int32_t a,int16_t b,int16x4_t v,const int lane)
//  int64_t vqdmlsls_lane_s32(int64_t a,int32_t b,int32x2_t v,const int lane)
//  int32x4_t vqdmlsl_high_lane_s16(int32x4_t a,int16x8_t b,int16x4_t v,const int lane)
//  int64x2_t vqdmlsl_high_lane_s32(int64x2_t a,int32x4_t b,int32x2_t v,const int lane)
//  int32x4_t vqdmlsl_laneq_s16(int32x4_t a,int16x4_t b,int16x8_t v,const int lane)
//  int64x2_t vqdmlsl_laneq_s32(int64x2_t a,int32x2_t b,int32x4_t v,const int lane)
//  int32_t vqdmlslh_laneq_s16(int32_t a,int16_t b,int16x8_t v,const int lane)
//  int64_t vqdmlsls_laneq_s32(int64_t a,int32_t b,int32x4_t v,const int lane)
//  int32x4_t vqdmlsl_high_laneq_s16(int32x4_t a,int16x8_t b,int16x8_t v,const int lane)
//  int64x2_t vqdmlsl_high_laneq_s32(int64x2_t a,int32x4_t b,int32x4_t v,const int lane)
/* clang-format on */
