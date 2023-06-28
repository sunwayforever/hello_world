// 2023-04-28 10:59
#include <mips_msa.h>
#include <msa_test.h>
// FADD.df1 F2 fadd_df(F,F) f3 + f Floating-Point Addition
// FDIV.df F fdiv_df(F,F) f / f Floating-Point Division
// FEXP2.df F fexp2_df(F,V4) Floating-Point Base 2 Exponentiation
// FLOG2.df F flog2_df(F,F) Floating-Point Base 2 Logarithm
//
// FMADD.df F fmadd_df(F,F) f + f * f Floating-Point Fused Multiply-Add
// FMSUB.df F fmsub_df(F,F) f - f * f Floating-Point Fused Multiply-Subtract
//
// FMAX.df F fmax_df(F,F) Floating-Point Maximum
// FMIN.df F fmin_df(F,F) Floating-Point Minimum
// FMAX_A.df F fmax_a_df(F,F) Floating-Point Maximum of Absolute Values
// FMIN_A.df F fmin_a_df(F,F) Floating-Point Minimum of Absolute Values
// FMUL.df F fmul_df(F,F) f * f Floating-Point Multiplication
// FRCP.df F frcp_df(F,F) Approximate Floating-Point Reciprocal
// FRINT.df F frint_df(F,F) Floating-Point Round to Integer
// FRSQRT.df F frsqrt_df(F,F) Approximate Floating-Point Reciprocal of Square
// Root FSQRT.df F fsqrt_df(F,F) Floating-Point Square Root FSUB.df F
// fsub_df(F,F) f - f Floating-Point Subtraction
TEST_CASE(test_fadd) {
    static struct {
        float a[4];
        float b[4];
        float r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {427.79, NAN, NAN, -999.94},
            {NAN, NAN, NAN, -882.979980},
        },
        {
            {-619.20, -413.47, 422.55, 160.51},
            {871.28, -660.33, 148.88, 905.13},
            {252.080017, -1073.800049, 571.429993, 1065.640015},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {182.75, -737.07, 165.68, 413.12},
            {-422.780029, -1708.540039, -16.380005, -265.419983},
        },
        {
            {20.28, -770.49, 949.17, 616.00},
            {647.00, -632.40, -967.88, -301.85},
            {667.280029, -1402.890015, -18.710022, 314.149994},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        b.i32 = __msa_ld_w((int32_t*)test_vec[i].b, 0);
        r.f32 = __msa_fadd_w(a.f32, b.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f32, check.f32);
    }
    return 0;
}

TEST_CASE(test_fmadd) {
    static struct {
        float c[4];
        float a[4];
        float b[4];
        float r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {427.79, NAN, NAN, -999.94},
            {NAN, NAN, NAN, -882.979980},
            {NAN, NAN, NAN, 883043.937500},
        },
        {
            {-619.20, -413.47, 422.55, 160.51},
            {871.28, -660.33, 148.88, 905.13},
            {252.080017, -1073.800049, 571.429993, 1065.640015},
            {219013.078125, 708648.937500, 85497.046875, 964703.250000},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {182.75, -737.07, 165.68, 413.12},
            {-422.780029, -1708.540039, -16.380005, -265.419983},
            {-77868.578125, 1258342.125000, -2895.899170, -110328.843750},
        },
        {
            {20.28, -770.49, 949.17, 616.00},
            {647.00, -632.40, -967.88, -301.85},
            {667.280029, -1402.890015, -18.710022, 314.149994},
            {431750.468750, 886417.187500, 19058.226562, -94210.179688},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 c, a, b, r, check;
        c.i32 = __msa_ld_w((int32_t*)test_vec[i].c, 0);
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        b.i32 = __msa_ld_w((int32_t*)test_vec[i].b, 0);
        r.f32 = __msa_fmadd_w(c.f32, a.f32, b.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f32, check.f32);
    }
    return 0;
}

TEST_CASE(test_frint) {
    static struct {
        float a[4];
        float r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {NAN, 657.000000, NAN, 117.000000},
        },
        {
            {-619.20, -413.47, 422.55, 160.51},
            {-619.000000, -413.000000, 423.000000, 161.000000},
        },
        {
            {-605.53, -971.47, -182.06, -678.54},
            {-606.000000, -971.000000, -182.000000, -679.000000},
        },
        {
            {20.28, -770.49, 949.17, 616.000000},
            {20.000000, -770.000000, 949.000000, 616.000000},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        r.f32 = __msa_frint_w(a.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f32, check.f32);
    }
    return 0;
}
