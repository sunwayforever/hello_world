// 2023-04-28 13:01
#include <mips_msa.h>
#include <msa_test.h>
// FCAF.df V fcaf_df(F,F) Floating-Point Quiet Compare Always False
// FCUN.df V fcun_df(F,F) Floating-Point Quiet Compare Unordered
// FCOR.df V fcor_df(F,F) Floating-Point Quiet Compare Ordered
// FCEQ.df V fceq_df(F,F) Floating-Point Quiet Compare Equal
// FCUNE.df V fcune_df(F,F) Floating-Point Quiet Compare Unordered or Not Equal
// FCUEQ.df V fcueq_df(F,F) Floating-Point Quiet Compare Unordered or Equal
// FCNE.df V fcne_df(F,F) Floating-Point Quiet Compare Not Equal
// FCLT.df V fclt_df(F,F) Floating-Point Quiet Compare Less Than
// FCULT.df V fcult_df(F,F) Floating-Point Quiet Compare Unordered or Less Than
// FCLE.df V fcle_df(F,F) Floating-Point Quiet Compare Less Than or Equal
// FCULE.df V fcule_df(F,F) Floating-Point Quiet Compare Unordered or Less Than
// or Equal

TEST_CASE(test_fcun) {
    static struct {
        float a[4];
        float b[4];
        int32_t r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {427.79, NAN, NAN, -999.94},
            {-1, -1, -1, 0},
        },
        {
            {-619.20, -413.47, 422.55, 160.51},
            {871.28, -660.33, 148.88, 905.13},
            {0, 0, 0, 0},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        b.i32 = __msa_ld_w((int32_t*)test_vec[i].b, 0);
        r.i32 = __msa_fcun_w(a.f32, b.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f32, check.f32);
    }
    return 0;
}

TEST_CASE(test_fcult) {
    static struct {
        float a[4];
        float b[4];
        int32_t r[4];
    } test_vec[] = {
        {
            {NAN, 656.90, NAN, 116.96},
            {427.79, NAN, NAN, -999.94},
            {-1, -1, -1, 0},
        },
        {
            {-619.20, -413.47, 422.55, 160.51},
            {871.28, -660.33, 148.88, 905.13},
            {-1, 0, 0, -1},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        m128 a, b, r, check;
        a.i32 = __msa_ld_w((int32_t*)test_vec[i].a, 0);
        b.i32 = __msa_ld_w((int32_t*)test_vec[i].b, 0);
        r.i32 = __msa_fcult_w(a.f32, b.f32);
        check.i32 = __msa_ld_w((int32_t*)test_vec[i].r, 0);
        ASSERT_CLOSE(r.f32, check.f32);
    }
    return 0;
}
