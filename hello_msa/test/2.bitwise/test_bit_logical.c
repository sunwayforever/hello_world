// 2023-04-27 16:47
#include <mips_msa.h>
#include <msa_test.h>
// AND.V V1 and_v(V,V) v2 & v Logical And
// ANDI.B V andi_b(V,K3) v & k4 Logical And Immediate
// OR.V V or_v(V,V) v | v Logical Or
// ORI.B V ori_b(V,K) v | k Logical Or Immediate
// NOR.V V nor_v(V,V) Logical Negated Or
// NORI.B V nori_b(V,K) Logical Negated Or Immediate
// XOR.V V xor_v(V,V) v ^ v Logical Or
// XORI.B V xori_b(V,K) v ^ k Logical Or Immediate
// BNEG.df V bneg_df(V,V) Bit Negate
// BNEGI.df V bnegi_df(V,K) Bit Negate Immediate
TEST_CASE(test_andv_b) {
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
            {137, 8, 176, 144, 133, 11, 193, 112, 12, 128, 72, 3, 128, 98, 97,
             37},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {100, 45, -92, 25, 125, 104, -111, -103, 37, 54, -20, 14, -83, -51,
             -59, 44},
            {100, 1, 32, 0, 109, 40, 17, 0, 36, 0, 36, 0, 40, 128, 197, 40},
        },
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i8 = __msa_ld_b(test_vec[i].a, 0);
        b.i8 = __msa_ld_b(test_vec[i].b, 0);
        r.u8 = __msa_and_v(a.u8, b.u8);
        check.i8 = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(r.u8, check.u8);
    }
    return 0;
}
