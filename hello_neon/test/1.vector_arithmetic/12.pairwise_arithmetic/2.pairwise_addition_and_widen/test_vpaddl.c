// 2023-04-18 13:39
#include <neon.h>
#include <neon_test.h>
// int16x4_t vpaddl_s8(int8x8_t a)
// int32x2_t vpaddl_s16(int16x4_t a)
// int64x1_t vpaddl_s32(int32x2_t a)
// uint16x4_t vpaddl_u8(uint8x8_t a)
// uint32x2_t vpaddl_u16(uint16x4_t a)
// uint64x1_t vpaddl_u32(uint32x2_t a)
//
// int16x8_t vpaddlq_s8(int8x16_t a)
// int32x4_t vpaddlq_s16(int16x8_t a)
// int64x2_t vpaddlq_s32(int32x4_t a)
// uint16x8_t vpaddlq_u8(uint8x16_t a)
// uint32x4_t vpaddlq_u16(uint16x8_t a)
// uint64x2_t vpaddlq_u32(uint32x4_t a)
// -------------------------------------------
// int16x4_t vpadal_s8(int16x4_t a,int8x8_t b)
//            ^^^^---pairwise add and accumulate, r[i]=a[i]+b[2*i]+b[2*i+1]
// int32x2_t vpadal_s16(int32x2_t a,int16x4_t b)
// int64x1_t vpadal_s32(int64x1_t a,int32x2_t b)
// uint16x4_t vpadal_u8(uint16x4_t a,uint8x8_t b)
// uint32x2_t vpadal_u16(uint32x2_t a,uint16x4_t b)
// uint64x1_t vpadal_u32(uint64x1_t a,uint32x2_t b)
//
// int16x8_t vpadalq_s8(int16x8_t a,int8x16_t b)
// int32x4_t vpadalq_s16(int32x4_t a,int16x8_t b)
// int64x2_t vpadalq_s32(int64x2_t a,int32x4_t b)
// uint16x8_t vpadalq_u8(uint16x8_t a,uint8x16_t b)
// uint32x4_t vpadalq_u16(uint32x4_t a,uint16x8_t b)
// uint64x2_t vpadalq_u32(uint64x2_t a,uint32x4_t b)

TEST_CASE(test_vpadal_s8) {
    static const struct {
        int16_t a[4];
        int8_t b[8];
        int16_t r[4];
    } test_vec[] = {
        {{-30161, 28803, 7944, -11953},
         {24, -114, 75, 41, -47, -102, 58, 60},
         {-30251, 28919, 7795, -11835}},
        {{20613, -1419, -15592, -26460},
         {99, -24, 126, 100, -26, 88, 13, 22},
         {20688, -1193, -15530, -26425}},
        {{-28446, -5242, -10833, -14404},
         {100, 7, -16, 53, -95, 43, 114, 38},
         {-28339, -5205, -10885, -14252}},
        {{-6277, -27872, -14934, 3371},
         {-83, -87, 113, -109, 2, 126, -87, -28},
         {-6447, -27868, -14806, 3256}},
        {{12302, -16945, -29947, 27012},
         {-109, 116, -97, 52, -97, 17, 91, 26},
         {12309, -16990, -30027, 27129}},
        {{31736, -23890, -9920, -4689},
         {-125, 32, -127, -123, -98, 42, 105, -84},
         {31643, -24140, -9976, -4668}},
        {{14682, 24681, -4668, 22473},
         {97, 104, -116, 1, 121, -25, 27, 113},
         {14883, 24566, -4572, 22613}},
        {{-13982, -23789, -15709, 9872},
         {-30, 17, -85, INT8_MIN, 60, 20, 44, -106},
         {-13995, -24002, -15629, 9810}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int16x4_t r = vpadal_s8(a, b);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
