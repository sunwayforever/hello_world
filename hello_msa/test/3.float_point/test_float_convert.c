// 2023-04-28 14:28
#include <mips_msa.h>
#include <msa_test.h>
// clang-format off
// FEXUPL.df1 F2 fexupl_df(G3) Left-Half Floating-Point Format Up-Convert
// FEXUPR.df F fexupr_df(G) Right-Half Floating-Point Format Up-Convert
// FEXDO.df G fexdo_df(F,F) Floating-Point Format Down-Convert
// FFINT_S.df F ffint_s_df(V4) Floating-Point Convert from Signed Integer
// FFINT_U.df F ffint_u_df(V) Floating-Point Convert from Unsigned Integer
// FFQL.df F ffql_df(W5) Left-Half Floating-Point Convert from Fixed-Point
// FFQR.df F ffqr_df(W) Right-Half Floating-Point Convert from Fixed-Point
// FTINT_S.df V ftint_s_df(V,V) Floating-Point Round and Convert to Signed Integer
// FTINT_U.df V ftint_u_df(V,V) Floating-Point Round and Convert to Unsigned Integer
// FTRUNC_S.df V ftrunc_s_df(F) Floating-Point Truncate and Convert to Signed Integer
// FTRUNC_U.df V ftrunc_u_df(F) Floating-Point Truncate and Convert to Unsigned Integer
// FTQ.df W ftq_df(F,F) Floating-Point Round and Convert to Fixed-Point
// clang-format on
TEST_CASE(test_fexupl) {
    static struct {
        float a[4];
        double r[2];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {NAN, 656.900024},
        },
        {
            {-619.20, -619.20, 422.55, 161.51},
            {-619.200012, -619.200012},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {-605.530029, -971.469971},
        },
        {
            {20.28, -770.49, 949.17, 616.00},
            {20.280001, -770.489990},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        /* TODO: fail with `-O3` */
        r.f64 = __msa_fexupr_d(a.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f64, check.f64);
    }
    return 0;
}
TEST_CASE(test_ftint_u) {
    static struct {
        float a[4];
        uint32_t r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {0, 657, 0, 117},
        },
        {
            {-619.20, -619.20, 422.5, 161.5},
            {0, 0, 422, 162},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {0, 0, 0, 0},
        },
        {
            {20.28, -770.49, 949.17, 616.00},
            {20, 0, 949, 616},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        r.u32 = __msa_ftint_u_w(a.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_EQUAL(r.u32, check.u32);
    }
    return 0;
}

TEST_CASE(test_ftrunc_s) {
    static struct {
        float a[4];
        int32_t r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {0, 656, 0, 116},
        },
        {
            {-619.20, -619.20, 422.55, 161.51},
            {-619, -619, 422, 161},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {-605, -971, -182, -678},
        },
        {
            {20.28, -770.49, 949.17, 616.00},
            {20, -770, 949, 616},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        r.i32 = __msa_ftrunc_s_w(a.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_EQUAL(r.i32, check.i32);
    }
    return 0;
}

TEST_CASE(test_ftq_h) {
    static struct {
        float a[4];
        float b[4];
        int16_t r[8];
    } test_vec[] = {
        {
            {0.15, -1.90, 0.51, 1.56},
            {0.85, -0.67, 0.44, 0.86},
            {27853, -21955, 14418, 28180, 4915, -32768, 16712, 32767},
        },
        {
            {-0.41, 0.91, 0.92, -0.55},
            {-0.41, 0.10, -1.69, -0.99},
            {-13435, 3277, -32768, -32440, -13435, 29819, 30147, -18022},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        b.i32 = __msa_ld_w((int32_t*)test_vec[i].b, 0);
        r.i16 = __msa_ftq_h(a.f32, b.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_EQUAL(r.i16, check.i16);
    }
    return 0;
}
