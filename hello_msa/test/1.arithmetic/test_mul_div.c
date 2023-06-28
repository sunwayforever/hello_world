// 2023-04-14 10:44
#include <mips_msa.h>
#include <msa_test.h>
// DIV_S.df V div_s_df(V,V) v / v Signed Divide
// DIV_U.df V div_u_df(V,V) v / v Unsigned Divide
// MULV.df V mulv_df(V,V) v * v Multiply
// MOD_S.df V mod_s_df(V,V) v % v Signed Remainder (Modulo)
// MOD_U.df V mod_u_df(V,V) v % v Unsigned Remainder (Modulo)
TEST_CASE(test_div_s_b) {
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
            {0, 0, 0, 0, 1, -1, 0, 0, 2, 0, 0, -1, 5, 0, -4, -1},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {1, -1, -1, 0, 0, 0, -1, 0, 0, -2, -1, 8, -1, 1, 0, 2},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 b = __msa_ld_b(test_vec[i].b, 0);
        v16i8 r = __msa_div_s_b(a, b);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
