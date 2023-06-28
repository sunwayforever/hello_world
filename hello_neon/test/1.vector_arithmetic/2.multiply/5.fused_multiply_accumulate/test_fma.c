// 2023-04-16 16:55
#include <neon.h>
#include <neon_test.h>

// float32x2_t vfma_f32(float32x2_t a,float32x2_t b,float32x2_t c)
//              ^^^ fused multply accumulate, r[i]=a[i]+b[i]*c[i], 但与 mla 不同的是, fma 只针对
//                  最终结果计算一次舍入, 而 mla 会针对 b*c 舍入一次后再针对最终结果再舍入一次, 所以
//                  fma 的精度更高
// float64x1_t vfma_f64(float64x1_t a,float64x1_t b,float64x1_t c)
//
// float32x4_t vfmaq_f32(float32x4_t a,float32x4_t b,float32x4_t c)
//                 ^--- 128-bit vector
// float64x2_t vfmaq_f64(float64x2_t a,float64x2_t b,float64x2_t c)
// -----------------------------------------------------------------------------------
// float32x2_t vfma_lane_f32(float32x2_t a,float32x2_t b,float32x2_t v,const int lane)
//                  ^^^^--- r[i]=a[i]+b[i]*v[lane]
// float64x1_t vfma_lane_f64(float64x1_t a,float64x1_t b,float64x1_t v,const int lane)
//
// float32x4_t vfmaq_lane_f32(float32x4_t a,float32x4_t b,float32x2_t v,const int lane)
// float64x2_t vfmaq_lane_f64(float64x2_t a,float64x2_t b,float64x1_t v,const int lane)
//
// float32x2_t vfma_laneq_f32(float32x2_t a,float32x2_t b,float32x4_t v,const int lane)
//                      ^--- lane 是 128-bit vector
// float64x1_t vfma_laneq_f64(float64x1_t a,float64x1_t b,float64x2_t v,const int lane)
// float32x4_t vfmaq_laneq_f32(float32x4_t a,float32x4_t b,float32x4_t v,const int lane)
// float64x2_t vfmaq_laneq_f64(float64x2_t a,float64x2_t b,float64x2_t v,const int lane)
// -----------------------------------------------------------------------------------
// float32_t vfmas_lane_f32(float32_t a,float32_t b,float32x2_t v,const int lane)
//               ^--- scalar, s 表示 single float, r=a+b*v[lane]
// float64_t vfmad_lane_f64(float64_t a,float64_t b,float64x1_t v,const int lane)
//               ^--- d 表示 double float
//
// float32_t vfmas_laneq_f32(float32_t a,float32_t b,float32x4_t v,const int lane)
// float64_t vfmad_laneq_f64(float64_t a,float64_t b,float64x2_t v,const int lane)
// -----------------------------------------------------------------------------------
// float32x2_t vfms_f32(float32x2_t a,float32x2_t b,float32x2_t c)
//                ^--- fused multiply subtract
// float64x1_t vfms_f64(float64x1_t a,float64x1_t b,float64x1_t c)
// float32x4_t vfmsq_f32(float32x4_t a,float32x4_t b,float32x4_t c)
// float64x2_t vfmsq_f64(float64x2_t a,float64x2_t b,float64x2_t c)
// -----------------------------------------------------------------------------------
// float32x2_t vfms_lane_f32(float32x2_t a,float32x2_t b,float32x2_t v,const int lane)
// float64x1_t vfms_lane_f64(float64x1_t a,float64x1_t b,float64x1_t v,const int lane)
// float32x4_t vfmsq_lane_f32(float32x4_t a,float32x4_t b,float32x2_t v,const int lane)
// float64x2_t vfmsq_lane_f64(float64x2_t a,float64x2_t b,float64x1_t v,const int lane)
// float32x2_t vfms_laneq_f32(float32x2_t a,float32x2_t b,float32x4_t v,const int lane)
// float64x1_t vfms_laneq_f64(float64x1_t a,float64x1_t b,float64x2_t v,const int lane)
// float32x4_t vfmsq_laneq_f32(float32x4_t a,float32x4_t b,float32x4_t v,const int lane)
// float64x2_t vfmsq_laneq_f64(float64x2_t a,float64x2_t b,float64x2_t v,const int lane)
// -----------------------------------------------------------------------------------
// float32_t vfmss_lane_f32(float32_t a,float32_t b,float32x2_t v,const int lane)
// float64_t vfmsd_lane_f64(float64_t a,float64_t b,float64x1_t v,const int lane)
// float32_t vfmss_laneq_f32(float32_t a,float32_t b,float32x4_t v,const int lane)
// float64_t vfmsd_laneq_f64(float64_t a,float64_t b,float64x2_t v,const int lane)
//
TEST_CASE(test_vfma_f32) {
    static const struct {
        float a[2];
        float b[2];
        float c[2];
        float r[2];
    } test_vec[] = {
        {{-418.49, -138.55},
         {-524.10, 787.36},
         {147.35, -958.10},
         {-77644.62, -754508.12}},
        {{-853.74, -358.34},
         {531.45, -127.29},
         {-403.29, -763.98},
         {-215182.22, 96888.67}},
        {{-14.18, -322.90},
         {334.11, 84.20},
         {798.51, 508.71},
         {266776.00, 42510.48}},
        {{-96.91, -205.64},
         {562.33, -123.00},
         {50.61, -890.35},
         {28362.61, 109307.41}},
        {{916.53, 885.37},
         {109.83, 977.63},
         {-333.79, 850.31},
         {-35743.63, 832173.94}},
        {{819.13, 247.72},
         {-288.24, -704.97},
         {35.08, 859.11},
         {-9292.33, -605399.00}},
        {{-663.07, 181.34},
         {-499.23, 868.39},
         {-945.95, 97.47},
         {471583.56, 84823.31}},
        {{-895.59, 39.87},
         {774.58, 438.52},
         {-875.93, 573.09},
         {-679373.44, 251351.30}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        float32x2_t c = vld1_f32(test_vec[i].c);
        float32x2_t r = vfma_f32(a, b, c);
        float32x2_t check = vld1_f32(test_vec[i].r);
        ASSERT_CLOSE(r, check);
    }
    return 0;
}

TEST_CASE(test_vfma_lane_f32) {
    static const struct {
        float a[2];
        float b[2];
        float v[2];
        float r0[2];
        float r1[2];
    } test_vec[] = {
        {{999.66, -447.69},
         {931.75, -966.11},
         {960.10, -428.65},
         {895572.81, -928009.88},
         {-398394.97, 413675.34}},
        {{-603.14, -528.38},
         {943.48, -556.98},
         {-474.92, 831.18},
         {-448680.66, 263992.56},
         {783598.56, -463479.00}},
        {{130.86, -707.32},
         {-905.88, -283.20},
         {-961.35, -421.24},
         {870998.56, 271547.00},
         {381723.75, 118587.85}},
        {{-41.10, 982.32},
         {-854.49, 700.89},
         {-621.88, 479.82},
         {531349.12, -434887.16},
         {-410042.50, 337283.38}},
        {{-753.75, -107.31},
         {907.27, -277.70},
         {-111.65, -801.86},
         {-102050.45, 30897.90},
         {-728257.25, 222569.22}},
        {{-102.95, -111.99},
         {-249.55, -171.20},
         {-78.10, -289.45},
         {19386.90, 13258.73},
         {72129.30, 49441.85}},
        {{400.15, 318.76},
         {182.18, 343.63},
         {761.79, 707.25},
         {139183.05, 262092.66},
         {129246.95, 243351.08}},
        {{174.81, -107.35},
         {999.93, 268.93},
         {609.45, -961.42},
         {609582.12, 163792.03},
         {-961177.88, -258662.02}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t v = vld1_f32(test_vec[i].v);
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        float32x2_t r0 = vfma_lane_f32(a, b, v, 0);
        float32x2_t r1 = vfma_lane_f32(a, b, v, 1);
        float32x2_t check0 = vld1_f32(test_vec[i].r0);
        float32x2_t check1 = vld1_f32(test_vec[i].r1);
        ASSERT_CLOSE(r0, check0);
        ASSERT_CLOSE(r1, check1);
    }
    return 0;
}

TEST_CASE(test_vfma_laneq_f32) {
    static const struct {
        float a[2];
        float b[2];
        float v[4];
        float r0[2];
        float r1[2];
        float r2[2];
        float r3[2];
    } test_vec[] = {
        {{847.69, -431.65},
         {-979.10, 993.21},
         {-730.77, -600.98, 473.02, -484.51},
         {716344.62, -726239.75},
         {589267.19, -597331.00},
         {-462286.16, 469376.53},
         {475231.44, -481651.84}},
        {{291.70, 380.29},
         {237.79, -819.95},
         {578.43, -865.16, 68.06, -671.12},
         {137836.56, -473903.38},
         {-205434.69, 709768.25},
         {16475.69, -55425.50},
         {-159293.92, 550665.12}},
        {{-36.36, 989.96},
         {39.43, -636.21},
         {308.73, -778.40, 707.42, 70.51},
         {12136.86, -195427.17},
         {-30728.67, 496215.84},
         {27857.21, -449077.72},
         {2743.85, -43869.21}},
        {{928.86, -117.77},
         {963.16, 928.78},
         {-848.84, 572.61, 967.36, 998.85},
         {-816639.88, -788503.44},
         {552443.88, 531710.94},
         {932651.25, 898346.88},
         {962981.19, 927594.12}},
        {{-859.04, 988.25},
         {992.06, -589.81},
         {-612.73, 465.08, -74.32, 678.97},
         {-608723.94, 362382.53},
         {460528.22, -273320.56},
         {-74588.94, 44822.93},
         {672719.94, -399475.03}},
        {{-154.63, -836.54},
         {859.02, -576.20},
         {-701.70, -72.92, -247.32, 261.94},
         {-602929.00, 403483.00},
         {-62794.37, 41179.96},
         {-212607.47, 141669.25},
         {224857.08, -151766.38}},
        {{-82.96, 792.10},
         {625.73, -774.23},
         {-986.29, 333.15, 296.28, 942.56},
         {-617234.19, 764407.38},
         {208378.98, -257142.61},
         {185308.31, -228596.75},
         {589705.06, -728966.12}},
        {{-784.62, 259.44},
         {871.35, -633.46},
         {-167.95, 838.70, -634.61, -26.99},
         {-147127.84, 106649.05},
         {730016.62, -531023.50},
         {-553752.00, 402259.50},
         {-24302.36, 17356.53}}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x4_t v = vld1q_f32(test_vec[i].v);
        float32x2_t a = vld1_f32(test_vec[i].a);
        float32x2_t b = vld1_f32(test_vec[i].b);
        float32x2_t r0 = vfma_laneq_f32(a, b, v, 0);
        float32x2_t r1 = vfma_laneq_f32(a, b, v, 1);
        float32x2_t r2 = vfma_laneq_f32(a, b, v, 2);
        float32x2_t r3 = vfma_laneq_f32(a, b, v, 3);
        float32x2_t check0 = vld1_f32(test_vec[i].r0);
        float32x2_t check1 = vld1_f32(test_vec[i].r1);
        float32x2_t check2 = vld1_f32(test_vec[i].r2);
        float32x2_t check3 = vld1_f32(test_vec[i].r3);
        ASSERT_CLOSE(r0, check0);
        ASSERT_CLOSE(r1, check1);
        ASSERT_CLOSE(r2, check2);
        ASSERT_CLOSE(r3, check3);
    }
    return 0;
}

TEST_CASE(test_vfmas_lane_f32) {
    static const struct {
        float a;
        float b;
        float v[2];
        float r0;
        float r1;
    } test_vec[] = {
        {-827.78, 859.69, {-743.58, -735.77}, -640076.06, -633361.94},
        {846.44, 758.06, {-780.68, -139.59}, -590955.81, -104971.15},
        {843.73, -745.56, {-314.55, 303.49}, 235359.62, -225426.27},
        {666.63, 325.58, {-872.69, 362.33}, -283463.78, 118634.02},
        {-229.72, 768.01, {-477.09, -601.98}, -366639.62, -462556.38},
        {-941.70, -969.77, {146.34, -117.39}, -142857.84, 112899.60},
        {-671.27, 103.73, {-267.47, 683.20}, -28415.93, 70197.07},
        {541.93, -484.44, {-358.32, 714.15}, 174126.47, -345420.91}};

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        float32x2_t v = vld1_f32(test_vec[i].v);
        float r0 = vfmas_lane_f32(test_vec[i].a, test_vec[i].b, v, 0);
        float r1 = vfmas_lane_f32(test_vec[i].a, test_vec[i].b, v, 1);
        ASSERT_CLOSE_SCALAR(r0, test_vec[i].r0);
        ASSERT_CLOSE_SCALAR(r1, test_vec[i].r1);
    }
    return 0;
}
