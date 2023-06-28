// 2023-04-20 11:56
#include <neon.h>
#include <neon_test.h>
// int32x2_t vcvt_s32_f32(float32x2_t a)
//            ^^^---默认 round to zero
// uint32x2_t vcvt_u32_f32(float32x2_t a)
// int32x2_t vcvtn_s32_f32(float32x2_t a)
//               ^---round to even
// uint32x2_t vcvtn_u32_f32(float32x2_t a)
// int32x2_t vcvtm_s32_f32(float32x2_t a)
//               ^---round to minus
// int32x2_t vcvtp_s32_f32(float32x2_t a)
//               ^---round to plus
// uint32x2_t vcvtp_u32_f32(float32x2_t a)
// int32x2_t vcvta_s32_f32(float32x2_t a)
//               ^---round to away
// uint32x2_t vcvta_u32_f32(float32x2_t a)
// int64x1_t vcvt_s64_f64(float64x1_t a)
// uint64x1_t vcvt_u64_f64(float64x1_t a)
// int64x1_t vcvtn_s64_f64(float64x1_t a)
// uint64x1_t vcvtn_u64_f64(float64x1_t a)
// int64x1_t vcvtm_s64_f64(float64x1_t a)
// uint64x1_t vcvtm_u64_f64(float64x1_t a)
// int64x1_t vcvtp_s64_f64(float64x1_t a)
// uint64x1_t vcvtp_u64_f64(float64x1_t a)
// int64x1_t vcvta_s64_f64(float64x1_t a)
// uint64x1_t vcvta_u64_f64(float64x1_t a)
// float32x2_t vcvt_f32_s32(int32x2_t a)
// float64x1_t vcvt_f64_s64(int64x1_t a)
// float64x1_t vcvt_f64_u64(uint64x1_t a)
// float32x2_t vcvt_n_f32_s32(int32x2_t a,const int n)
//                  ^---num of fraction bits
// float32x2_t vcvt_n_f32_u32(uint32x2_t a,const int n)
// float64x1_t vcvt_n_f64_s64(int64x1_t a,const int n)
// float64x1_t vcvt_n_f64_u64(uint64x1_t a,const int n)
// int32x2_t vcvt_n_s32_f32(float32x2_t a,const int n)
// uint32x2_t vcvt_n_u32_f32(float32x2_t a,const int n)
// int64x1_t vcvt_n_s64_f64(float64x1_t a,const int n)
// uint64x1_t vcvt_n_u64_f64(float64x1_t a,const int n)
//
// int32x4_t vcvtq_s32_f32(float32x4_t a)
// uint32x4_t vcvtq_u32_f32(float32x4_t a)
// int32x4_t vcvtnq_s32_f32(float32x4_t a)
// uint32x4_t vcvtnq_u32_f32(float32x4_t a)
// int32x4_t vcvtmq_s32_f32(float32x4_t a)
// uint32x2_t vcvtm_u32_f32(float32x2_t a)
// uint32x4_t vcvtmq_u32_f32(float32x4_t a)
// int32x4_t vcvtpq_s32_f32(float32x4_t a)
// uint32x4_t vcvtpq_u32_f32(float32x4_t a)
// int32x4_t vcvtaq_s32_f32(float32x4_t a)
// uint32x4_t vcvtaq_u32_f32(float32x4_t a)
// int64x2_t vcvtq_s64_f64(float64x2_t a)
// uint64x2_t vcvtq_u64_f64(float64x2_t a)
// int64x2_t vcvtnq_s64_f64(float64x2_t a)
// uint64x2_t vcvtnq_u64_f64(float64x2_t a)
// int64x2_t vcvtmq_s64_f64(float64x2_t a)
// uint64x2_t vcvtmq_u64_f64(float64x2_t a)
// int64x2_t vcvtpq_s64_f64(float64x2_t a)
// uint64x2_t vcvtpq_u64_f64(float64x2_t a)
// int64x2_t vcvtaq_s64_f64(float64x2_t a)
// uint64x2_t vcvtaq_u64_f64(float64x2_t a)
// float32x4_t vcvtq_f32_s32(int32x4_t a)
// float32x4_t vcvtq_f32_u32(uint32x4_t a)
// float64x2_t vcvtq_f64_s64(int64x2_t a)
// float64x2_t vcvtq_f64_u64(uint64x2_t a)
// float32x4_t vcvtq_n_f32_s32(int32x4_t a,const int n)
// float32x4_t vcvtq_n_f32_u32(uint32x4_t a,const int n)
// float64x2_t vcvtq_n_f64_s64(int64x2_t a,const int n)
// float64x2_t vcvtq_n_f64_u64(uint64x2_t a,const int n)
// int32x4_t vcvtq_n_s32_f32(float32x4_t a,const int n)
// uint32x4_t vcvtq_n_u32_f32(float32x4_t a,const int n)
// int64x2_t vcvtq_n_s64_f64(float64x2_t a,const int n)
// uint64x2_t vcvtq_n_u64_f64(float64x2_t a,const int n)
// ------------------------------------------
// scalar:
// int32_t vcvts_s32_f32(float32_t a)
//             ^---scalar
// uint32_t vcvts_u32_f32(float32_t a)
// int32_t vcvtns_s32_f32(float32_t a)
// uint32_t vcvtns_u32_f32(float32_t a)
// int32_t vcvtms_s32_f32(float32_t a)
// uint32_t vcvtms_u32_f32(float32_t a)
// int32_t vcvtps_s32_f32(float32_t a)
// uint32_t vcvtps_u32_f32(float32_t a)
// int32_t vcvtas_s32_f32(float32_t a)
// int64_t vcvtd_s64_f64(float64_t a)
// uint32_t vcvtas_u32_f32(float32_t a)
// uint64_t vcvtd_u64_f64(float64_t a)
// int64_t vcvtnd_s64_f64(float64_t a)
// uint64_t vcvtnd_u64_f64(float64_t a)
// int64_t vcvtmd_s64_f64(float64_t a)
// uint64_t vcvtmd_u64_f64(float64_t a)
// int64_t vcvtpd_s64_f64(float64_t a)
// uint64_t vcvtpd_u64_f64(float64_t a)
// int64_t vcvtad_s64_f64(float64_t a)
// uint64_t vcvtad_u64_f64(float64_t a)
// int32_t vcvts_n_s32_f32(float32_t a,const int n)
// uint32_t vcvts_n_u32_f32(float32_t a,const int n)
// float32_t vcvts_f32_s32(int32_t a)
// float32_t vcvts_f32_u32(uint32_t a)
// float64_t vcvtd_f64_s64(int64_t a)
// float64_t vcvtd_f64_u64(uint64_t a)
// float32_t vcvts_n_f32_s32(int32_t a,const int n)
// float32_t vcvts_n_f32_u32(uint32_t a,const int n)
// float64_t vcvtd_n_f64_s64(int64_t a,const int n)
// float64_t vcvtd_n_f64_u64(uint64_t a,const int n)
// int64_t vcvtd_n_s64_f64(float64_t a,const int n)
// uint64_t vcvtd_n_u64_f64(float64_t a,const int n)
// ------------------------------------
// f16, f32, f64 之间转换:
// float16x4_t vcvt_f16_f32(float32x4_t a)
// float16x8_t vcvt_high_f16_f32(float16x4_t r,float32x4_t a)
// float32x2_t vcvt_f32_f64(float64x2_t a)
// float32x4_t vcvt_high_f32_f64(float32x2_t r,float64x2_t a)
// float32x4_t vcvt_f32_f16(float16x4_t a)
// float32x4_t vcvt_high_f32_f16(float16x8_t a)
// float64x2_t vcvt_f64_f32(float32x2_t a)
// float64x2_t vcvt_high_f64_f32(float32x4_t a)
// float32x2_t vcvtx_f32_f64(float64x2_t a)
// float32_t vcvtxd_f32_f64(float64_t a)
// float32x4_t vcvtx_high_f32_f64(float32x2_t r,float64x2_t a)

TEST_CASE(test_vcvt_s32_f32) {
    static const struct {
        float32_t a[2];
        int32_t r[2];
    } test_vec[] = {
        {{396.15, -246.90}, {396, -246}},   {{241.51, 602.56}, {241, 602}},
        {{-106.85, -566.67}, {-106, -566}}, {{463.44, 539.86}, {463, 539}},
        {{-550.41, 982.91}, {-550, 982}},   {{499.92, -727.55}, {499, -727}},
        {{-713.41, 713.10}, {-713, 713}},   {{-998.69, -409.99}, {-998, -409}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        int32x2_t r = vcvt_s32_f32(a);
        int32x2_t check = vld1_s32(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vcvta_s32_f32) {
    static const struct {
        float32_t a[2];
        int32_t r[2];
    } test_vec[] = {
        {{396.15, -246.90}, {396, -247}},   {{241.5, 602.56}, {242, 603}},
        {{-106.85, -566.67}, {-107, -567}}, {{463.44, 539.86}, {463, 540}},
        {{-550.41, 982.91}, {-550, 983}},   {{499.92, -727.55}, {500, -728}},
        {{-713.41, 713.10}, {-713, 713}},   {{-998.69, -409.99}, {-999, -410}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        int32x2_t r = vcvta_s32_f32(a);
        int32x2_t check = vld1_s32(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
