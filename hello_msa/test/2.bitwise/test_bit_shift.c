// 2023-04-27 16:59
#include <mips_msa.h>
#include <msa_test.h>
// SLL.df V sll_df(V,V) v << v Shift Left
// SLLI.df V slli_df(V,K) v << k Shift Left Immediate
// SRA.df V sra_df(V,V) v >> v Shift Right Arithmetic
// SRAI.df V srai_df(V,K) v >> k Shift Right Arithmetic Immediate
// SRAR.df V srar_df(V,V) Shift Right Arithmetic with Rounding
// SRARI.df V srari_df(V,K) Shift Right Arithmetic with Rounding Immediate
// SRL.df V srl_df(V,V) v >> v Shift Right
// SRLI.df V srli_df(V,K) v >> k Shift Right Immediate
// SRLR.df V srlr_df(V,V) Shift Right with Rounding
// SRLRI.df V srlri_df(V,K) Shift Right with Rounding Immediate

TEST_CASE(test_sll_h) {
    struct {
        int16_t a[8];
        int16_t b[8];
        int16_t r[8];
    } test_vec[] = {
        {
            {22332, -2389, -6176, 24298, -30833, -3392, 7263, 1769},
            {-11, -12, 535, 14, 15, -10006, -26926, -19506},
            {-6272, 27312, -4096, -32768, -32768, 0, 29052, 16384},
        },

        {
            {24162, 31020, -13216, -28236, -844, -30766, -24430, 3898},
            {9, 11, 16, 2895, 9, -9, -16, 2356},
            {-15360, 24576, -13216, 0, 26624, -5888, -24430, -3168},
        },

        {
            {7438, -17736, 2143, -16698, 2818, -21698, 29388, 22965},
            {-10, -11, -15, 12, 10, 27061, 13, -13277},
            {17280, 22272, 4286, 24576, 2048, 26560, -32768, -12888},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v8i16 a = __msa_ld_h(test_vec[i].a, 0);
        v8i16 b = __msa_ld_h(test_vec[i].b, 0);
        v8i16 r = __msa_sll_h(a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_sra_h) {
    struct {
        int16_t a[8];
        int16_t b[8];
        int16_t r[8];
    } test_vec[] = {
        {
            {22332, -2389, -6176, 24298, -30833, -3392, 7263, 1769},
            {-11, -12, 535, 14, 15, -10006, -26926, -19506},
            {697, -150, -49, 1, -1, -4, 1815, 0},
        },
        {
            {24162, 31020, -13216, -28236, -844, -30766, -24430, 3898},
            {9, 11, 16, 2895, 9, -9, -16, 2356},
            {47, 15, -13216, -1, -2, -241, -24430, 243},
        },
        {
            {7438, -17736, 2143, -16698, 2818, -21698, 29388, 22965},
            {-10, -11, -15, 12, 10, 27061, 13, -13277},
            {116, -555, 1071, -5, 2, -679, 3, 2870},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v8i16 a = __msa_ld_h(test_vec[i].a, 0);
        v8i16 b = __msa_ld_h(test_vec[i].b, 0);
        v8i16 r = __msa_sra_h(a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_srl_h) {
    struct {
        int16_t a[8];
        int16_t b[8];
        int16_t r[8];
    } test_vec[] = {
        {
            {22332, -2389, -6176, 24298, -30833, -3392, 7263, 1769},
            {-11, -12, 535, 14, 15, -10006, -26926, -19506},
            {697, 3946, 463, 1, 1, 60, 1815, 0},
        },
        {
            {24162, 31020, -13216, -28236, -844, -30766, -24430, 3898},
            {9, 11, 16, 2895, 9, -9, -16, 2356},
            {47, 15, -13216, 1, 126, 271, -24430, 243},
        },
        {
            {7438, -17736, 2143, -16698, 2818, -21698, 29388, 22965},
            {-10, -11, -15, 12, 10, 27061, 13, -13277},
            {116, 1493, 1071, 11, 2, 1369, 3, 2870},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v8i16 a = __msa_ld_h(test_vec[i].a, 0);
        v8i16 b = __msa_ld_h(test_vec[i].b, 0);
        v8i16 r = __msa_srl_h(a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_srlr_h) {
    struct {
        int16_t a[8];
        int16_t b[8];
        int16_t r[8];
    } test_vec[] = {
        {
            {22332, -2389, -6176, 24298, -30833, -3392, 7263, 1769},
            {-11, -12, 535, 14, 15, -10006, -26926, -19506},
            {698, 3947, 464, 1, 1, 61, 1816, 0},
        },
        {
            {24162, 31020, -13216, -28236, -844, -30766, -24430, 3898},
            {9, 11, 16, 2895, 9, -9, -16, 2356},
            {47, 15, -13216, 1, 126, 272, -24430, 244},
        },
        {
            {7438, -17736, 2143, -16698, 2818, -21698, 29388, 22965},
            {-10, -11, -15, 12, 10, 27061, 13, -13277},
            {116, 1494, 1072, 12, 3, 1370, 4, 2871},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v8i16 a = __msa_ld_h(test_vec[i].a, 0);
        v8i16 b = __msa_ld_h(test_vec[i].b, 0);
        v8i16 r = __msa_srlr_h(a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
