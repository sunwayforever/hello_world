// 2023-04-19 13:55
#include <neon.h>
#include <neon_test.h>
// int8x8_t vqshl_s8(int8x8_t a,int8x8_t b)
// int16x4_t vqshl_s16(int16x4_t a,int16x4_t b)
// int32x2_t vqshl_s32(int32x2_t a,int32x2_t b)
// int64x1_t vqshl_s64(int64x1_t a,int64x1_t b)
// uint8x8_t vqshl_u8(uint8x8_t a,int8x8_t b)
// uint16x4_t vqshl_u16(uint16x4_t a,int16x4_t b)
// uint32x2_t vqshl_u32(uint32x2_t a,int32x2_t b)
// uint64x1_t vqshl_u64(uint64x1_t a,int64x1_t b)
//
// int8x16_t vqshlq_s8(int8x16_t a,int8x16_t b)
// int16x8_t vqshlq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vqshlq_s32(int32x4_t a,int32x4_t b)
// int64x2_t vqshlq_s64(int64x2_t a,int64x2_t b)
// uint8x16_t vqshlq_u8(uint8x16_t a,int8x16_t b)
// uint16x8_t vqshlq_u16(uint16x8_t a,int16x8_t b)
// uint32x4_t vqshlq_u32(uint32x4_t a,int32x4_t b)
// uint64x2_t vqshlq_u64(uint64x2_t a,int64x2_t b)
// -----------------------------------------------
// int8_t vqshlb_s8(int8_t a,int8_t b)
//             ^---scalar
// int32_t vqshls_s32(int32_t a,int32_t b)
// int16_t vqshlh_s16(int16_t a,int16_t b)
// int64_t vqshld_s64(int64_t a,int64_t b)
// uint8_t vqshlb_u8(uint8_t a,int8_t b)
// uint16_t vqshlh_u16(uint16_t a,int16_t b)
// uint32_t vqshls_u32(uint32_t a,int32_t b)
// uint64_t vqshld_u64(uint64_t a,int64_t b)
// -----------------------------------------------
// int8x8_t vqshl_n_s8(int8x8_t a,const int n)
// int16x4_t vqshl_n_s16(int16x4_t a,const int n)
// int32x2_t vqshl_n_s32(int32x2_t a,const int n)
// int64x1_t vqshl_n_s64(int64x1_t a,const int n)
// uint8x8_t vqshl_n_u8(uint8x8_t a,const int n)
// uint16x4_t vqshl_n_u16(uint16x4_t a,const int n)
// uint32x2_t vqshl_n_u32(uint32x2_t a,const int n)
// uint64x1_t vqshl_n_u64(uint64x1_t a,const int n)
//
// int8x16_t vqshlq_n_s8(int8x16_t a,const int n)
// int16x8_t vqshlq_n_s16(int16x8_t a,const int n)
// int32x4_t vqshlq_n_s32(int32x4_t a,const int n)
// int64x2_t vqshlq_n_s64(int64x2_t a,const int n)
// uint8x16_t vqshlq_n_u8(uint8x16_t a,const int n)
// uint16x8_t vqshlq_n_u16(uint16x8_t a,const int n)
// uint32x4_t vqshlq_n_u32(uint32x4_t a,const int n)
// uint64x2_t vqshlq_n_u64(uint64x2_t a,const int n)
// -------------------------------------------------
// int8_t vqshlb_n_s8(int8_t a,const int n)
// int16_t vqshlh_n_s16(int16_t a,const int n)
// int32_t vqshls_n_s32(int32_t a,const int n)
// int64_t vqshld_n_s64(int64_t a,const int n)
// uint8_t vqshlb_n_u8(uint8_t a,const int n)
// uint16_t vqshlh_n_u16(uint16_t a,const int n)
// uint32_t vqshls_n_u32(uint32_t a,const int n)
// uint64_t vqshld_n_u64(uint64_t a,const int n)
// ------------------------------------------------
// uint8_t vqshlub_n_s8(int8_t a,const int n)
// uint16_t vqshluh_n_s16(int16_t a,const int n)
// uint32_t vqshlus_n_s32(int32_t a,const int n)
// uint64_t vqshlud_n_s64(int64_t a,const int n)
// ------------------------------------------------
// uint8x8_t vqshlu_n_s8(int8x8_t a,const int n)
// uint16x4_t vqshlu_n_s16(int16x4_t a,const int n)
// uint32x2_t vqshlu_n_s32(int32x2_t a,const int n)
// uint64x1_t vqshlu_n_s64(int64x1_t a,const int n)
//
// uint8x16_t vqshluq_n_s8(int8x16_t a,const int n)
// uint16x8_t vqshluq_n_s16(int16x8_t a,const int n)
// uint32x4_t vqshluq_n_s32(int32x4_t a,const int n)
// uint64x2_t vqshluq_n_s64(int64x2_t a,const int n)

TEST_CASE(test_vqshl_s16) {
    static const struct {
        int16_t a[4];
        int16_t b[4];
        int16_t r[4];
    } test_vec[] = {
        {{629, -1930, -203, -57},
         {24, 25, 9, 19},
         {INT16_MAX, INT16_MIN, INT16_MIN, INT16_MIN}},
        {{341, -2, -51, 320},
         {15, 9, 26, 24},
         {INT16_MAX, -1024, INT16_MIN, INT16_MAX}},
        {{-9, -1567, -31654, -66},
         {25, 13, 19, 0},
         {INT16_MIN, INT16_MIN, INT16_MIN, -66}},
        {{-3881, 1242, 43, 24},
         {3, 31, 2, 27},
         {-31048, INT16_MAX, 172, INT16_MAX}},
        {{15, 991, -4, -31},
         {23, 10, 20, 16},
         {INT16_MAX, INT16_MAX, INT16_MIN, INT16_MIN}},
        {{38, -758, 1, 1012},
         {9, 27, 30, 22},
         {19456, INT16_MIN, INT16_MAX, INT16_MAX}},
        {{974, 63, -79, 8},
         {30, 3, 14, 29},
         {INT16_MAX, 504, INT16_MIN, INT16_MAX}},
        {{-97, 7, 935, -2},
         {13, 9, 30, 25},
         {INT16_MIN, 3584, INT16_MAX, INT16_MIN}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int16x4_t r = vqshl_s16(a, b);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vqshl_u64) {
    static const struct {
        uint64_t a[1];
        int64_t b[1];
        uint64_t r[1];
    } test_vec[] = {{{11758907}, {53}, {UINT64_MAX}},
                    {{201}, {39}, {110500918591488}},
                    {{10353}, {60}, {UINT64_MAX}},
                    {{16865279727}, {54}, {UINT64_MAX}},
                    {{298491154210}, {26}, {UINT64_MAX}},
                    {{45}, {59}, {UINT64_MAX}},
                    {{158}, {54}, {2846274964498153472}},
                    {{415649}, {33}, {3570397723230208}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        uint64x1_t a = vld1_u64(test_vec[i].a);
        int64x1_t b = vld1_s64(test_vec[i].b);
        uint64x1_t r = vqshl_u64(a, b);
        uint64x1_t check = vld1_u64(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }

    return 0;
}
