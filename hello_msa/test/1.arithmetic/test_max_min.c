// 2023-04-27 16:07
#include <mips_msa.h>
#include <msa_test.h>
// MAX_A.df V max_a_df(V,V) Maximum of Absolute Values
// MIN_A.df V min_a_df(V,V) Minimum of Absolute Values
// MAX_S.df V max_s_df(V,V) Signed Maximum
// MAXI_S.df V maxi_s_df(V,K) Signed Immediate Maximum
// MAX_U.df V max_u_df(V,V) Unsigned Maximum
// MAXI_U.df V maxi_u_df(V,K) Unsigned Immediate Maximum
// MIN_S.df V min_s_df(V,V) Signed Maximum
// MINI_S.df V mini_s_df(V,K) Signed Immediate Maximum
// MIN_U.df V min_u_df(V,V) Unsigned Maximum
// MINI_U.df V mini_u_df(V,K) Unsigned Immediate Maximum
TEST_CASE(test_max_s_b) {
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
            {-51, 43, -16, -79, -75, 79, -47, 120, 28, -124, 88, 115, -23, 119,
             125, 109},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {127, 45, 104, 25, 125, 104, 121, 6, 37, 54, 37, 112, 106, -51, -35,
             120},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_max_s_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}

TEST_CASE(test_maxi_s_b) {
    struct {
        int8_t a[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {10, 10, 10, 10, 10, 10, 10, 115, 28, 10, 10, 10, 10, 10, 125, 109},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {127, 10, 104, 10, 109, 57, 121, 10, 10, 10, 37, 112, 106, 10, 10,
             120},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 r = __msa_maxi_s_b(a, 10);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
