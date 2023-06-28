// 2023-04-18 14:55
#include <neon.h>
#include <neon_test.h>
// int16_t vaddlv_s8(int8x8_t a)
//             ^---widen
// int32_t vaddlv_s16(int16x4_t a)
// int64_t vaddlv_s32(int32x2_t a)
// uint16_t vaddlv_u8(uint8x8_t a)
// uint32_t vaddlv_u16(uint16x4_t a)
// uint64_t vaddlv_u32(uint32x2_t a)
//
// int16_t vaddlvq_s8(int8x16_t a)
// int32_t vaddlvq_s16(int16x8_t a)
// int64_t vaddlvq_s32(int32x4_t a)
// uint16_t vaddlvq_u8(uint8x16_t a)
// uint32_t vaddlvq_u16(uint16x8_t a)
// uint64_t vaddlvq_u32(uint32x4_t a)

TEST_CASE(test_vaddlv_s8) {
    static const struct {
        int8_t a[8];
        int16_t r;
    } test_vec[] = {
        {{-78, 109, 126, 107, -63, -93, -101, 9}, 16},
        {{77, -96, -92, -35, -11, 95, 28, -74}, -108},
        {{20, -83, 94, -114, -15, 30, 119, -22}, 29},
        {{-61, 72, 20, 67, 26, -4, -83, -52}, -15},
        {{106, 43, 55, 43, -50, -46, 52, 27}, 230},
        {{115, -40, -8, 104, 56, 20, 30, 76}, 353},
        {{-63, 125, -38, -77, -101, 82, -99, 94}, -77},
        {{-102, -78, -94, -76, -82, 79, INT8_MIN, 24}, -457}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int16_t r = vaddlv_s8(a);
        ASSERT_EQUAL_SCALAR(r, test_vec[i].r);
    }
    return 0;
}
