// 2023-04-28 16:17
#include <mips_msa.h>
#include <msa_test.h>
// clang-format off
// CLE_S.df V cle_s_df(V,V) v <= v Compare Less-Than-or-Equal Signed
// CLEI_S.df V clei_s_df(V,K) v <= k Compare Less-Than-or-Equal Signed Immediate
// CLE_U.df V cle_u_df(V,V) v <= v Compare Less-Than-or-Equal Unsigned
// CLEI_U.df V clei_u_df(V,K) v <= k Compare Less-Than-or-Equal Unsigned Immediate
// CLT_S.df V clt_s_df(V,V) v < v Compare Less-Than Signed
// CLTI_S.df V clti_s_df(V,K) v < k Compare Less-Than Signed Immediate
// CLT_U.df V clt_u_df(V,V) v < v Compare Less-Than Unsigned
// CLTI_U.df V clti_u_df(V,K) v < k Compare Less-Than Unsigned Immediate
// clang-format on

TEST_CASE(test_clt_s_b) {
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
            {0, -1, 0, 0, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, 0},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {0, -1, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0},
        },
        {
            {-98, 45, 51, 11, 102, -84, 17, -53, -10, 21, 122, 97, -73, 88, -39,
             -36},
            {-38, -74, -28, 87, 30, 118, -16, 10, -114, -59, -21, 59, -110, -80,
             103, 49},
            {-1, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1},
        },
        {
            {-34, -102, 60, 68, 71, 78, 15, 33, 126, -90, -63, 53, -2, -101, 18,
             -116},
            {4, -12, 120, 34, 106, 104, 44, 96, 96, -3, -57, -13, -83, 47, 36,
             -117},
            {-1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, 0},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_clt_s_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
