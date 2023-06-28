// 2023-04-27 15:34
#include <mips_msa.h>
#include <msa_test.h>
// DOTP_S.df V dotp_s_df(W,W) Signed Dot Product
// DOTP_U.df V dotp_u_df(W,W) Unsigned Dot Product
// DPADD_S.df V dpadd_s_df(V,W,W) Signed Dot Product Add
// DPADD_U.df V dpadd_u_df(V,W,W) Unsigned Dot Product Add
// DPSUB_S.df V dpsub_s_df(V,W,W) Signed Dot Product Subtract
// DPSUB_U.df V dpsub_u_df(V,W,W) Unsigned Dot Product Subtract
TEST_CASE(test_dotp_s) {
    struct {
        int8_t a[16];
        int8_t b[16];
        int16_t r[8];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {5597, 8958, 1096, 16667, 15892, -14951, 234, -11832},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {10585, -9418, 19553, -14049, -7652, 828, -4004, 7345},
        },

    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v8i16 r = __msa_dotp_s_h(a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_dpadd_s) {
    struct {
        int16_t c[8];
        int8_t a[16];
        int8_t b[16];
        int16_t r[8];
    } test_vec[] = {
        {
            {5597, 8958, 1096, 16667, 15892, -14951, 234, -11832},
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {-103, 43, -76, -98, -75, 79, -61, 120, 14, -125, 88, 115, -23, 119,
             -31, -73},
            {11194, 17916, 2192, -32202, 31784, -29902, 468, -23664},
        },
        {
            {10585, -9418, 19553, -14049, -7652, 828, -4004, 7345},
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {21170, -18836, -26430, -28098, -15304, 1656, -8008, 14690},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v8i16 c = __msa_ld_h(test_vec[i].c, 0);
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v8i16 r = __msa_dpadd_s_h(c, a, b);
        v8i16 check = __msa_ld_h(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
