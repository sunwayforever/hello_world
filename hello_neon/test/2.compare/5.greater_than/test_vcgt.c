// 2023-04-18 17:18
#include <neon.h>
#include <neon_test.h>
// uint8x8_t vcgt_s8(int8x8_t a,int8x8_t b)
// uint16x4_t vcgt_s16(int16x4_t a,int16x4_t b)
// uint32x2_t vcgt_s32(int32x2_t a,int32x2_t b)
// uint64x1_t vcgt_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vcgt_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vcgt_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vcgt_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vcgt_u64(uint64x1_t a,uint64x1_t b)
// uint32x2_t vcgt_f32(float32x2_t a,float32x2_t b)
// uint64x1_t vcgt_f64(float64x1_t a,float64x1_t b)
//
// uint8x16_t vcgtq_s8(int8x16_t a,int8x16_t b)
// uint16x8_t vcgtq_s16(int16x8_t a,int16x8_t b)
// uint32x4_t vcgtq_s32(int32x4_t a,int32x4_t b)
// uint64x2_t vcgtq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vcgtq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vcgtq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vcgtq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vcgtq_u64(uint64x2_t a,uint64x2_t b)
// uint32x4_t vcgtq_f32(float32x4_t a,float32x4_t b)
// uint64x2_t vcgtq_f64(float64x2_t a,float64x2_t b)
// --------------------------------------------------
// uint64_t vcgtd_s64(int64_t a,int64_t b)
// uint64_t vcgtd_u64(uint64_t a,uint64_t b)
// uint32_t vcgts_f32(float32_t a,float32_t b)
// uint64_t vcgtd_f64(float64_t a,float64_t b)
// uint8x8_t vcgtz_s8(int8x8_t a)
// uint8x16_t vcgtzq_s8(int8x16_t a)
// uint16x4_t vcgtz_s16(int16x4_t a)
// uint16x8_t vcgtzq_s16(int16x8_t a)
// uint32x2_t vcgtz_s32(int32x2_t a)
// uint32x4_t vcgtzq_s32(int32x4_t a)
// uint64x1_t vcgtz_s64(int64x1_t a)
// uint64x2_t vcgtzq_s64(int64x2_t a)
// uint32x2_t vcgtz_f32(float32x2_t a)
// uint32x4_t vcgtzq_f32(float32x4_t a)
// uint64x1_t vcgtz_f64(float64x1_t a)
// uint64x2_t vcgtzq_f64(float64x2_t a)
// uint64_t vcgtzd_s64(int64_t a)
// uint32_t vcgtzs_f32(float32_t a)
// uint64_t vcgtzd_f64(float64_t a)

TEST_CASE(test_vcgt_s8) {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        uint8_t r[8];
    } test_vec[] = {
        {
            {-6, -44, 28, 95, -61, 16, 5, -50},
            {-6, -88, 27, 22, -61, 34, -63, -8},
            {0, 255, 255, 255, 0, 0, 255, 0},

        },
        {
            {114, -100, -127, 77, -6, -19, -116, -77},
            {-63, -88, -127, 19, -71, -122, -31, -77},
            {255, 0, 0, 255, 255, 255, 0, 0},

        },
        {
            {-104, -76, -104, -83, -97, -5, 119, 96},
            {-104, -78, -104, -110, -97, 37, -77, 96},
            {0, 255, 0, 255, 0, 0, 255, 0},

        },
        {
            {122, 3, -37, 110, -46, -119, 113, 100},
            {122, -42, 18, 82, -46, -119, -99, 106},
            {0, 255, 0, 255, 0, 0, 255, 0},

        },
        {
            {-20, 100, -81, 122, 1, 48, -33, 81},
            {-20, 100, -65, 122, 41, 48, -33, -93},
            {0, 0, 0, 0, 0, 0, 0, 255},

        },
        {
            {-55, -4, -112, 122, -27, -6, -53, -46},
            {120, -4, 113, 122, -27, -6, -53, -47},
            {0, 0, 0, 0, 0, 0, 0, 255},

        },
        {
            {29, 33, -101, -106, -76, -34, 76, 126},
            {29, -35, -94, 12, -88, -17, -34, 32},
            {0, 255, 0, 0, 255, 0, 255, 255},

        },
        {
            {123, -18, 47, 48, 25, 24, -82, 6},
            {123, 10, -53, 48, -23, 24, -82, 6},
            {0, 0, 255, 0, 255, 0, 0, 0},
        }};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        uint8x8_t r = vcgt_s8(a, b);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
