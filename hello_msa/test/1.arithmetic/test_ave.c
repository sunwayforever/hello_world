// 2023-04-27 15:07
#include <mips_msa.h>
#include <msa_test.h>
// AVE_S.df V ave_s_df(V,V) Signed Average
// AVE_U.df V ave_u_df(V,V) Unsigned Average
// AVER_S.df V aver_s_df(V,V) Signed Average with Rounding
// AVER_U.df V aver_u_df(V,V) Unsigned Average with Rounding
TEST_CASE(test_ave_s_b) {
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
            {-77, 25, -46, -89, -98, -11, -54, 117, 21, -125, 35, -1, -74, 48,
             47, 18},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {113, -1, 6, 15, 117, 80, 5, -49, 8, -37, 8, 63, 11, -73, -47, 82},

        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_ave_s_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_aver_s_b) {
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
            {-77, 26, -46, -88, -98, -11, -54, 118, 21, -124, 36, -1, -73, 49,
             47, 18},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {114, -1, 6, 16, 117, 81, 5, -48, 9, -37, 9, 63, 12, -72, -47, 82},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_aver_s_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
