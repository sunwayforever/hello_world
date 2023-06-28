// 2023-04-28 16:44
#include <mips_msa.h>
#include <msa_test.h>
// clang-format off
// CFCMSA N cfcmsa(K2) Copy from MSA Control Register
// CTCMSA void ctcmsa(N,K) Copy to MSA Control Register
// LD.df V ld_df(*V) v = *pv Load Vector
// LDI.df V ldi_df(K) v = (V){k7,…,k} Load Immediate
// MOVE.V V move_v(V) v = v Vector to Vector Move
// SPLAT.df V splat_df(V,N) v = (V){v[n8],…,v[n]} Replicate Vector Element
// SPLATI.df V splati_df(V,K) v = (V){v[k],…,v[k]} Replicate Vector Element Immediate
// FILL.df V fill_df(N) v = (V){n,…,n} Fill Vector from GPR
// INSERT.df V insert_df(V,K,N) v[k] = n Insert GPR to Vector Element
// INSVE.df V insve_df(V,K,V) v[k] = v[0] Insert Vector element 0 to Vector Element
// COPY_S.df N copy_s_df(V,K) n = v[k] Copy element to GPR Signed
// COPY_U.df N copy_u_df(V,K) n = v[k] Copy element to GPR Unsigned
// ST.df V st_df(*V,V) *pv = v Store Vector
// clang-format on

/* NOTE: 通过 cfcmsa/ctcmsa 读写 msacsr, 以控制 rounding mode */
TEST_CASE(test_splat_b) {
    struct {
        int8_t a[16];
        int8_t r[16];
    } test_vec[] = {
        {
            {-51, 8, -16, -79, -121, -101, -47, 115, 28, -124, -17, -117, -124,
             -22, 125, 109},
            {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
        },
        {
            {INT8_MAX, -47, 104, 6, 109, 57, 121, 6, -20, INT8_MIN, 37, 112,
             106, -94, -35, 120},
            {-47, -47, -47, -47, -47, -47, -47, -47, -47, -47, -47, -47, -47,
             -47, -47, -47},
        },

    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        v16i8 a = __msa_ld_b(test_vec[i].a, 0);
        v16i8 r = __msa_splat_b(a, 1);
        v16i8 check = __msa_ld_b(test_vec[i].r, 0);
        ASSERT_EQUAL(check, r);
    }
    return 0;
}
