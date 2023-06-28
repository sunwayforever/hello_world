// 2023-04-18 18:07
#include <neon.h>
#include <neon_test.h>
/* NOTE: vtst 与 vceq 没有关系: vtst 用来比较 a 被 b mask 后是否不为零 */
// uint8x8_t vtst_s8(int8x8_t a,int8x8_t b)
// uint16x4_t vtst_s16(int16x4_t a,int16x4_t b)
// uint32x2_t vtst_s32(int32x2_t a,int32x2_t b)
// uint64x1_t vtst_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vtst_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vtst_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vtst_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vtst_u64(uint64x1_t a,uint64x1_t b)
//
// uint8x16_t vtstq_s8(int8x16_t a,int8x16_t b)
// uint16x8_t vtstq_s16(int16x8_t a,int16x8_t b)
// uint32x4_t vtstq_s32(int32x4_t a,int32x4_t b)
// uint64x2_t vtstq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vtstq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vtstq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vtstq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vtstq_u64(uint64x2_t a,uint64x2_t b)
// -----------------------------------------------
// uint8x8_t vtst_p8(poly8x8_t a,poly8x8_t b)
// uint64x1_t vtst_p64(poly64x1_t a,poly64x1_t b)
// uint8x16_t vtstq_p8(poly8x16_t a,poly8x16_t b)
// uint64x2_t vtstq_p64(poly64x2_t a,poly64x2_t b)
// -----------------------------------------------
// uint64_t vtstd_s64(int64_t a,int64_t b)
// uint64_t vtstd_u64(uint64_t a,uint64_t b)

TEST_CASE(test_vtst_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        uint8_t r[8];
    } test_vec[] = {
        {{-42, -92, 8, 20, 123, -127, -20, 74},
         {-48, 90, -3, 68, 104, 126, -103, 100},
         {UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX,
          UINT8_MAX}},
        {{58, 21, 103, -65, -93, 33, -117, -9},
         {-59, -108, 11, 64, 92, -9, -117, 44},
         {0, UINT8_MAX, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX}},
        {{63, 126, -127, -6, 27, -11, 62, 121},
         {-88, -127, 126, -55, -94, 10, -63, 104},
         {UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, 0, 0, UINT8_MAX}},
        {{-68, -24, -49, -62, -97, -18, 26, -84},
         {80, 23, -40, 108, -86, -14, -27, 83},
         {UINT8_MAX, 0, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, 0}},
        {{61, 38, 6, -98, -66, -40, -65, 22},
         {-62, 83, -40, 97, 65, 39, 64, -111},
         {0, UINT8_MAX, 0, 0, 0, 0, 0, UINT8_MAX}},
        {{-58, 89, -108, 108, 54, 45, 86, -32},
         {57, -57, 78, -109, -13, -46, -87, -75},
         {0, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX, 0, 0, UINT8_MAX}},
        {{111, -11, 43, 98, -40, 14, -127, 31},
         {103, -115, -44, -99, -97, -107, 126, -40},
         {UINT8_MAX, UINT8_MAX, 0, 0, UINT8_MAX, UINT8_MAX, 0, UINT8_MAX}},
        {{-106, 26, 41, 17, -7, 33, 39, -32},
         {22, -8, 90, -18, 6, 82, 13, 109},
         {UINT8_MAX, UINT8_MAX, UINT8_MAX, 0, 0, 0, UINT8_MAX, UINT8_MAX}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        uint8x8_t r = vtst_s8(a, b);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }

    return 0;
}
