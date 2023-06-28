// 2023-04-27 17:37
#include <mips_msa.h>
#include <msa_test.h>
// BCLR.df5 V bclr_df(V,V) Bit Clear
// BCLRI.df V bclri_df(V,K) Bit Clear Immediate
//
// BINSL.df V binsl_df(V,V) Insert Left of Bit Position
// BINSLI.df V binsli_df(V,K) Insert Left of Immediate Bit Position
// BINSR.df V binsr_df(V,V) Insert Right of Bit Position
// BINSRI.df V binsri_df(V,K) Insert Right of Immediate Bit Position
//
// BMNZ.V V bmnz_v(V,V,V) Bit Move If Not Zero
// BMNZI.B V bmnzi_b(V,V,K) Bit Move If Not Zero Immediate
// BMZ.V V bmz_v(V,V,V) Bit Move If Zero
// BMZI.B V bmzi_b(V,V,K) Bit Move If Zero Immediate
//
// BSEL.V V bsel_v(V,V) Bit Select
// BSELI.B V bseli_b(V,K) Bit Select Immediate
//
// BSET.df V bset_df(V,V) Bit Set
// BSETI.df V bseti_df(V,K) Bit Set Immediate
//
// NLOC.df V nloc_df(V,V) Leading One Bits Count
// NLZC.df V nlzc_df(V,V) Leading Zero Bits Count
//
// PCNT.df V pcnt_df(V,V) Population (Bits Set to 1) Count

TEST_CASE(test_bclr) {
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
            {205, 0, 224, 177, 135, 27, 209, 114, 28, 132, 238, 131, 132, 106,
             125, 109},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {111, 209, 104, 4, 77, 56, 121, 4, 204, 128, 37, 48, 74, 130, 221,
             104},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_bclr_b(a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}

TEST_CASE(test_bset) {
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
            {207, 8, 240, 241, 167, 155, 217, 115, 92, 140, 239, 139, 134, 234,
             127, 237},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {127, 241, 120, 6, 109, 57, 123, 6, 236, 192, 53, 112, 106, 162,
             253, 120},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_bset_b(a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}

TEST_CASE(test_binsl) {
    struct {
        int8_t c[16];
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {205, 0, 224, 177, 135, 27, 209, 114, 28, 132, 238, 131, 132, 106,
             125, 109},
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {205, 0, 240, 177, 135, 155, 209, 114, 28, 132, 238, 131, 132, 234,
             125, 109},

        },
        {
            {111, 209, 104, 4, 77, 56, 121, 4, 204, 128, 37, 48, 74, 130, 221,
             104},
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {127, 209, 104, 4, 109, 56, 121, 4, 236, 128, 37, 112, 106, 162,
             221, 120},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 c, a, b, r, check;
        c.i8 = __msa_ld_b(test_vec[i].c, 0);
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_binsl_b(c.u8, a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}

TEST_CASE(test_bmnz_v) {
    struct {
        int8_t c[16];
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {205, 0, 224, 177, 135, 27, 209, 114, 28, 132, 238, 131, 132, 106,
             125, 109},
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {205, 8, 240, 177, 135, 27, 209, 114, 28, 132, 238, 131, 132, 106,
             125, 109},
        },
        {
            {111, 209, 104, 4, 77, 56, 121, 4, 204, 128, 37, 48, 74, 130, 221,
             104},
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {111, 209, 104, 4, 109, 56, 121, 4, 236, 128, 37, 48, 106, 130, 221,
             104},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 c, a, b, r, check;
        c.i8 = __msa_ld_b(test_vec[i].c, 0);
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_bmnz_v(c.u8, a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}

TEST_CASE(test_bsel_v) {
    struct {
        int8_t c[16];
        int8_t a[16];
        int8_t b[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {205, 0, 224, 177, 135, 27, 209, 114, 28, 132, 238, 131, 132, 106,
             125, 109},
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {137, 8, 176, 144, 133, 139, 193, 113, 12, 128, 73, 11, 128, 226,
             97, 37},
        },
        {
            {111, 209, 104, 4, 77, 56, 121, 4, 204, 128, 37, 48, 74, 130, 221,
             104},
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {116, 1, 32, 2, 109, 41, 17, 2, 36, 0, 36, 64, 40, 160, 197, 56},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 c, a, b, r, check;
        c.i8 = __msa_ld_b(test_vec[i].c, 0);
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_bsel_v(c.u8, a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}

TEST_CASE(test_nloc) {
    struct {
        int8_t a[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {2, 0, 4, 1, 1, 1, 2, 0, 0, 1, 3, 1, 1, 3, 0, 0},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 2, 0},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 r = __msa_nloc_b(a);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_nlzc) {
    struct {
        int8_t a[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {0, 4, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {1, 0, 1, 5, 1, 2, 1, 5, 0, 0, 2, 1, 1, 0, 0, 1},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 r = __msa_nlzc_b(a);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
