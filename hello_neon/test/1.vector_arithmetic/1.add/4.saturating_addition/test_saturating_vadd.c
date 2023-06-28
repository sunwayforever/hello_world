// 2023-04-14 10:44
#include <neon.h>
#include <neon_test.h>

// int8x8_t vqadd_s8(int8x8_t a,int8x8_t b)
//           ^--- q 表示 saturating
// int16x4_t vqadd_s16(int16x4_t a,int16x4_t b)
// int32x2_t vqadd_s32(int32x2_t a,int32x2_t b)
// int64x1_t vqadd_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vqadd_u8(uint8x8_t a,uint8x8_t b)
// uint16x4_t vqadd_u16(uint16x4_t a,uint16x4_t b)
// uint32x2_t vqadd_u32(uint32x2_t a,uint32x2_t b)
// uint64x1_t vqadd_u64(uint64x1_t a,uint64x1_t b)
//
// int8x16_t vqaddq_s8(int8x16_t a,int8x16_t b)
//                ^--- q 表示 128-bit vector
// int16x8_t vqaddq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vqaddq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vqaddq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vqaddq_u8(uint8x16_t a,uint8x16_t b)
// uint16x8_t vqaddq_u16(uint16x8_t a,uint16x8_t b)
// uint32x4_t vqaddq_u32(uint32x4_t a,uint32x4_t b)
// uint64x2_t vqaddq_u64(uint64x2_t a,uint64x2_t b)
// -------------------------------------------------
// int8_t vqaddb_s8(int8_t a,int8_t b)
//             ^--- 计算 scalar 而不是 vector, b 表示 int8
// int16_t vqaddh_s16(int16_t a,int16_t b)
//              ^--- h 表示 HI (half int), int16_t
// int32_t vqadds_s32(int32_t a,int32_t b)
//              ^--- SI, int32_t
// int64_t vqaddd_s64(int64_t a,int64_t b)
//              ^--- DI, int64_t
// uint8_t vqaddb_u8(uint8_t a,uint8_t b)
// uint16_t vqaddh_u16(uint16_t a,uint16_t b)
// uint32_t vqadds_u32(uint32_t a,uint32_t b)
// uint64_t vqaddd_u64(uint64_t a,uint64_t b)
// -------------------------------------------------
// int8x8_t vuqadd_s8(int8x8_t a,uint8x8_t b)
//           ^--- b 是 unsigned int
// int8x16_t vuqaddq_s8(int8x16_t a,uint8x16_t b)
// int16x4_t vuqadd_s16(int16x4_t a,uint16x4_t b)
// int16x8_t vuqaddq_s16(int16x8_t a,uint16x8_t b)
// int32x2_t vuqadd_s32(int32x2_t a,uint32x2_t b)
// int32x4_t vuqaddq_s32(int32x4_t a,uint32x4_t b)
// int64x1_t vuqadd_s64(int64x1_t a,uint64x1_t b)
// int64x2_t vuqaddq_s64(int64x2_t a,uint64x2_t b)
// int8_t vuqaddb_s8(int8_t a,uint8_t b)
// int16_t vuqaddh_s16(int16_t a,uint16_t b)
// int32_t vuqadds_s32(int32_t a,uint32_t b)
// int64_t vuqaddd_s64(int64_t a,uint64_t b)
// -------------------------------------------------
// uint8x8_t vsqadd_u8(uint8x8_t a,int8x8_t b)
//            ^--- a 是 unsigned int
// uint8x16_t vsqaddq_u8(uint8x16_t a,int8x16_t b)
// uint16x4_t vsqadd_u16(uint16x4_t a,int16x4_t b)
// uint16x8_t vsqaddq_u16(uint16x8_t a,int16x8_t b)
// uint32x2_t vsqadd_u32(uint32x2_t a,int32x2_t b)
// uint32x4_t vsqaddq_u32(uint32x4_t a,int32x4_t b)
// uint64x1_t vsqadd_u64(uint64x1_t a,int64x1_t b)
// uint64x2_t vsqaddq_u64(uint64x2_t a,int64x2_t b)
// --------------------------------------------------
// uint8_t vsqaddb_u8(uint8_t a,int8_t b)
//               ^--- scalar
// uint16_t vsqaddh_u16(uint16_t a,int16_t b)
// uint32_t vsqadds_u32(uint32_t a,int32_t b)
// uint64_t vsqaddd_u64(uint64_t a,int64_t b)
//
static int test_vqadd_s8() {
    static const struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{-88, -39, -126, -86, -106, -49, 70, 124},
         {64, -91, 64, 47, 38, 91, -80, -15},
         {-24, INT8_MIN, -62, -39, -68, 42, -10, 109}},
        {{-18, -104, 112, 14, 66, 72, -83, -71},
         {-106, -119, 15, -10, 78, 70, INT8_MAX, -9},
         {-124, INT8_MIN, INT8_MAX, 4, INT8_MAX, INT8_MAX, 44, -80}},
        {{31, 1, -95, -74, -48, -25, 50, 16},
         {-116, 114, 64, -78, -51, -16, -93, -69},
         {-85, 115, -31, INT8_MIN, -99, -41, -43, -53}},
        {{-119, 19, -54, -53, 92, 119, -124, -14},
         {0, -109, -23, 79, -39, 104, 70, -7},
         {-119, -90, -77, 26, 53, INT8_MAX, -54, -21}},
        {{105, -25, -81, 58, -50, -31, 74, 90},
         {83, -118, 12, 32, 123, -80, -37, 4},
         {INT8_MAX, INT8_MIN, -69, 90, 73, -111, 37, 94}},
        {{-61, -91, -49, 31, 29, 83, 18, 29},
         {-25, -5, 108, -64, 99, -78, -71, -52},
         {-86, -96, 59, -33, INT8_MAX, 5, -53, -23}},
        {{-103, 104, 6, 103, 73, 81, -63, -100},
         {-37, -50, -68, 86, 126, -104, 90, 65},
         {INT8_MIN, 54, -62, INT8_MAX, INT8_MAX, -23, 27, -35}},
        {{61, 41, 97, 90, 125, 115, 120, 100},
         {110, -28, 36, -47, -105, -34, -99, 48},
         {INT8_MAX, 13, INT8_MAX, 43, 20, 81, 21, INT8_MAX}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vqadd_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vuqadd_s8) {
    static const struct {
        int8_t a[8];
        uint8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{84, 57, -62, -25, -68, 124, -101, 76},
         {225, 164, 56, 22, 32, 114, 246, 1},
         {INT8_MAX, INT8_MAX, -6, -3, -36, INT8_MAX, INT8_MAX, 77}},
        {{94, 81, 64, 37, 41, 123, 94, 108},
         {60, 73, 74, 85, 157, 66, 152, 241},
         {INT8_MAX, INT8_MAX, INT8_MAX, 122, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MAX}},
        {{123, 90, -40, 55, -41, 115, -125, -72},
         {23, 187, 206, 56, 45, 196, 57, 139},
         {INT8_MAX, INT8_MAX, INT8_MAX, 111, 4, INT8_MAX, -68, 67}},
        {{21, 122, -80, 63, -11, 14, -85, 49},
         {87, 245, 135, 244, 55, 31, 229, 178},
         {108, INT8_MAX, 55, INT8_MAX, 44, 45, INT8_MAX, INT8_MAX}},
        {{122, -67, -23, 81, 49, 108, 9, 72},
         {39, 216, 128, 85, 156, 186, 224, 178},
         {INT8_MAX, INT8_MAX, 105, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX,
          INT8_MAX}},
        {{52, -111, -15, 41, -97, -100, 90, -9},
         {145, 225, 235, 201, 1, 209, 123, 123},
         {INT8_MAX, 114, INT8_MAX, INT8_MAX, -96, 109, INT8_MAX, 114}},
        {{-114, 101, -52, -65, -47, -43, 8, -7},
         {173, 136, 78, 74, 66, 46, 252, 118},
         {59, INT8_MAX, 26, 9, 19, 3, INT8_MAX, 111}},
        {{-65, -19, -97, 95, -119, -6, 86, 26},
         {219, 65, 227, 220, 18, 95, 87, 161},
         {INT8_MAX, 46, INT8_MAX, INT8_MAX, -101, 89, INT8_MAX, INT8_MAX}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        uint8x8_t b = vld1_u8(test_vec[i].b);
        int8x8_t r = vuqadd_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);

        ASSERT_EQUAL(r, check);
    }

    return 0;
}

TEST_CASE(test_vsqadd_u8) {
    static const struct {
        uint8_t a[8];
        int8_t b[8];
        uint8_t r[8];
    } test_vec[] = {
        {{208, 12, 182, 206, 138, 82, 138, 210},
         {70, -34, -51, 0, 26, -3, 12, 116},
         {UINT8_MAX, 0, 131, 206, 164, 79, 150, UINT8_MAX}},
        {{13, 174, 176, 99, 171, 252, 208, 63},
         {-4, 58, 119, 3, -104, 51, -69, 105},
         {9, 232, UINT8_MAX, 102, 67, UINT8_MAX, 139, 168}},
        {{63, 113, 55, 202, 196, 193, 156, 10},
         {-97, 106, 10, -71, 103, 23, 46, 116},
         {0, 219, 65, 131, UINT8_MAX, 216, 202, 126}},
        {{197, 222, 216, 112, 219, 168, 176, 215},
         {-29, 39, -38, 123, 91, -107, -28, -102},
         {168, UINT8_MAX, 178, 235, UINT8_MAX, 61, 148, 113}},
        {{6, 27, 100, 202, 220, 1, 212, 124},
         {107, -33, 53, -46, -10, 99, 70, -69},
         {113, 0, 153, 156, 210, 100, UINT8_MAX, 55}},
        {{66, 30, 43, 29, 199, 219, 244, 170},
         {3, -50, 37, 94, 99, 10, -8, 105},
         {69, 0, 80, 123, UINT8_MAX, 229, 236, UINT8_MAX}},
        {{37, 93, 52, 2, 94, 8, 126, 201},
         {-25, -77, -101, -35, 23, -31, -104, 89},
         {12, 16, 0, 0, 117, 0, 22, UINT8_MAX}},
        {{0, 196, 118, 199, 159, 106, 113, 162},
         {56, -106, 0, -101, -96, -7, 4, -58},
         {56, 90, 118, 98, 63, 99, 117, 104}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint8x8_t a = vld1_u8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        uint8x8_t r = vsqadd_u8(a, b);
        uint8x8_t check = vld1_u8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }

    return 0;
}

TEST_CASE(test_vqaddb_s8) {
    static const struct {
        int8_t a;
        int8_t b;
        int8_t r;
    } test_vec[] = {{-47, 10, -37},       {58, 109, INT8_MAX},
                    {-66, 31, -35},       {INT8_MAX, -3, 124},
                    {88, 75, INT8_MAX},   {32, 124, INT8_MAX},
                    {-95, -49, INT8_MIN}, {-102, 38, -64}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8_t r = vqaddb_s8(test_vec[i].a, test_vec[i].b);
        assert(r == test_vec[i].r);
    }
    return 0;
}

TEST_CASE(test_vuqaddb_s8) {
    static const struct {
        int8_t a;
        uint8_t b;
        int8_t r;
    } test_vec[] = {{63, 186, INT8_MAX}, {46, 228, INT8_MAX},
                    {4, 92, 96},         {80, 144, INT8_MAX},
                    {-91, 184, 93},      {-82, 209, INT8_MAX},
                    {71, 212, INT8_MAX}, {126, 232, INT8_MAX}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8_t r = vuqaddb_s8(test_vec[i].a, test_vec[i].b);
        assert(r == test_vec[i].r);
    }

    return 0;
}

TEST_CASE(test_vsqaddb_u8) {
    static const struct {
        uint8_t a;
        int8_t b;
        uint8_t r;
    } test_vec[] = {
        {235, -3, 232}, {50, -75, 0},        {154, -12, 142},
        {74, 115, 189}, {171, INT8_MIN, 43}, {221, 110, UINT8_MAX},
        {153, 73, 226}, {123, 86, 209},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint8_t r = vsqaddb_u8(test_vec[i].a, test_vec[i].b);
        assert(r == test_vec[i].r);
    }
    return 0;
}
