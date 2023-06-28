// 2023-04-28 16:58
#include <mips_msa.h>
#include <msa_test.h>

// clang-format off
// ILVEV.df V ilvev_df(V,V) Interleave Even
// ILVOD.df V ilvod_df(V,V) Interleave Odd
// ILVL.df V ilvl_df(V,V) Interleave Left
// ILVR.df V ilvr_df(V,V) Interleave Right
// PCKEV.df V pckev_df(V,V) Pack Even Elements
// PCKOD.df V pckod_df(V,V) Pack Odd Elements
// SHF.df V shf_df(V,K) Set Shuffle
// SLD.df V sld_df(V,N) Element Slide
// SLDI.df V sldi_df(V,K) Element Slide Immediate
// VSHF.df V vshf_df(V,V,V) Vector shuffle
// clang-format on

TEST_CASE(test_ilvev_b) {
    struct {
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {-103, -51, -76, -16, -75, -121, -61, -47, 14, 28, 88, -17, -23,
             -124, -31, 125},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_ilvev_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_ilvl_b) {
    struct {
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {14, 28, -125, -124, 88, -17, 115, -117, -23, -124, 119, -22, -31,
             125, -73, 109},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_ilvl_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_pckev_b) {
    struct {
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {-103, -76, -75, -61, 14, 88, -23, -31, -51, -16, -121, -47, 28,
             -17, -124, 125},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_pckev_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_vshf_b) {
    struct {
        int8_t c[16];
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {-103, -76, -75, -61, 14, 88, -23, -31, -51, -16, -121, -47, 28,
             -17, -124, 125},
            {-103, -75, 14, -23, -51, -121, 28, -124, -103, -76, -75, -61, 14,
             88, -23, -31},

        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 c = __msa_ld_b(test_vec[i].c, 0);
        v16i8 r = __msa_vshf_b(c, a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_sld_b) {
    struct {
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119, -31,
             -73, -51},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_sld_b(a, b, 1);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
