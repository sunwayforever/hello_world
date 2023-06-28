// 2023-04-16 20:47
#include <neon.h>
#include <neon_test.h>

/* clang-format off */
// vqdmul                            h:
//  ^---qd 表示 saturating doubling, ^--h 表示 high part
//
// r[i]=(a[i]*b[i]*2)>>(esize/2)
//
// int16x4_t vqdmulh_s16(int16x4_t a,int16x4_t b)
// int32x2_t vqdmulh_s32(int32x2_t a,int32x2_t b)
//
// int16x8_t vqdmulhq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vqdmulhq_s32(int32x4_t a,int32x4_t b)
// -------------------------------------------------
// int16_t vqdmulhh_s16(int16_t a,int16_t b)
//                ^--- scalar, HI(int16)
// int32_t vqdmulhs_s32(int32_t a,int32_t b)
//                ^--- SI(int32)
// -------------------------------------------------
// int16x4_t vqrdmulh_s16(int16x4_t a,int16x4_t b)
//             ^--- rounding
// int32x2_t vqrdmulh_s32(int32x2_t a,int32x2_t b)
//
// int16x8_t vqrdmulhq_s16(int16x8_t a,int16x8_t b)
// int32x4_t vqrdmulhq_s32(int32x4_t a,int32x4_t b)
// -------------------------------------------------
// int16_t vqrdmulhh_s16(int16_t a,int16_t b)
//          ^^^---^^ staturating doubling, high part, HI
// int32_t vqrdmulhs_s32(int32_t a,int32_t b)
// -------------------------------------------------
// int32x4_t vqdmull_s16(int16x4_t a,int16x4_t b)
//                 ^--- widen
// int64x2_t vqdmull_s32(int32x2_t a,int32x2_t b)
// -------------------------------------------------
// int32_t vqdmullh_s16(int16_t a,int16_t b)
//          ^^---^^--- staturating doubling, widen, HI
// int64_t vqdmulls_s32(int32_t a,int32_t b)
// -------------------------------------------------
// int32x4_t vqdmull_high_s16(int16x8_t a,int16x8_t b)
//            ^^---^-^^^^ staturating doubling, widen, high (使用 vector 的后一半元素)
// int64x2_t vqdmull_high_s32(int32x4_t a,int32x4_t b)
/* clang-format on */

TEST_CASE(test_vqdmulh_s16) {
    static const struct {
        int16_t a[4];
        int16_t b[4];
        int16_t r[4];
    } test_vec[] = {
        {{10007, 28883, -16203, -25505},
         {-28965, -965, -21451, -19467},
         {-8846, -851, 10607, 15152}},
        {{297, -28727, 26792, 21146},
         {-2097, 13945, 6034, -22205},
         {-20, -12226, 4933, -14330}},
        {{5694, -3302, 31190, -20080},
         {-13560, 15789, -23944, -24080},
         {-2357, -1592, -22791, 14756}},
        {{-17757, 19504, -13790, -3682},
         {6337, 21287, 27183, 28413},
         {-3435, 12670, -11440, -3193}},
        {{6017, 22369, -3696, -26615},
         {-18755, 13781, -14759, -810},
         {-3444, 9407, 1664, 657}},
        {{1664, -23992, -6191, -28013},
         {-17665, 12006, -7388, -23140},
         {-898, -8791, 1395, 19782}},
        {{-262, -29955, 1775, -21469},
         {-1860, 5601, -18498, 15890},
         {14, -5121, -1003, -10411}},
        {{23230, -28704, 29505, 16417},
         {1837, 21103, 3050, -6921},
         {1302, -18486, 2746, -3468}},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int16x4_t r = vqdmulh_s16(a, b);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vqdmull_s16) {
    static const struct {
        int16_t a[4];
        int16_t b[4];
        int32_t r[4];
    } test_vec[] = {
        {{INT16_C(31681), INT16_C(13027), -INT16_C(13937), -INT16_C(20674)},
         {INT16_C(10302), -INT16_C(18422), INT16_C(4806), -INT16_C(12487)},
         {INT32_C(652755324), -INT32_C(479966788), -INT32_C(133962444),
          INT32_C(516312476)}},
        {{-INT16_C(13071), INT16_C(28436), INT16_C(8073), -INT16_C(13812)},
         {-INT16_C(4168), INT16_C(8843), INT16_C(11236), -INT16_C(23047)},
         {INT32_C(108959856), INT32_C(502919096), INT32_C(181416456),
          INT32_C(636650328)}},
        {{-INT16_C(8794), INT16_C(14039), INT16_C(5542), -INT16_C(6939)},
         {-INT16_C(4291), INT16_C(925), -INT16_C(10750), -INT16_C(3117)},
         {INT32_C(75470108), INT32_C(25972150), -INT32_C(119153000),
          INT32_C(43257726)}},
        {{-INT16_C(6238), INT16_C(11106), INT16_C(28167), -INT16_C(16394)},
         {-INT16_C(32418), INT16_C(17122), -INT16_C(9299), INT16_C(21479)},
         {INT32_C(404446968), INT32_C(380313864), -INT32_C(523849866),
          -INT32_C(704253452)}},
        {{-INT16_C(16712), INT16_C(24457), INT16_C(28627), INT16_C(4419)},
         {-INT16_C(8098), INT16_C(24596), -INT16_C(6217), INT16_C(22868)},
         {INT32_C(270667552), INT32_C(1203088744), -INT32_C(355948118),
          INT32_C(202107384)}},
        {{-INT16_C(18737), -INT16_C(10619), INT16_C(31525), -INT16_C(31851)},
         {INT16_C(30716), -INT16_C(22075), -INT16_C(21421), INT16_C(3069)},
         {-INT32_C(1151051384), INT32_C(468828850), -INT32_C(1350594050),
          -INT32_C(195501438)}},
        {{-INT16_C(31126), INT16_C(15722), -INT16_C(20747), INT16_C(21582)},
         {INT16_C(25486), INT16_C(17844), INT16_C(2122), INT16_C(6559)},
         {-INT32_C(1586554472), INT32_C(561086736), -INT32_C(88050268),
          INT32_C(283112676)}},
        {{INT16_C(9407), -INT16_C(6929), -INT16_C(31329), -INT16_C(25753)},
         {INT16_C(11516), INT16_C(20293), INT16_C(17112), INT16_C(16987)},
         {INT32_C(216662024), -INT32_C(281220394), -INT32_C(1072203696),
          -INT32_C(874932422)}},

    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int32x4_t r = vqdmull_s16(a, b);
        int32x4_t check = vld1q_s32(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vqrdmulh_s16) {
    static const struct {
        int16_t a[4];
        int16_t b[4];
        int16_t r[4];
    } test_vec[] = {
        {{24408, 8011, -30441, -30215},
         {4356, 11308, 3238, -19917},
         {3245, 2765, -3008, 18365}},
        {{11964, 20417, -7014, 9797},
         {3004, 6963, -19145, -28864},
         {1097, 4338, 4098, -8630}},
        {{-29676, 11183, -22507, 6580},
         {-7751, 24645, 30957, -21998},
         {7020, 8411, -21263, -4417}},
        {{-11354, 16889, 16055, 29799},
         {-26039, -32625, -12465, 25360},
         {9022, -16815, -6107, 23062}},
        {{-16549, 28814, 17255, 8329},
         {-12508, 4480, -28089, -4421},
         {6317, 3939, -14791, -1124}},
        {{-19354, 7471, -26894, 15249},
         {8240, -32580, -13072, 19427},
         {-4867, -7428, 10729, 9041}},
        {{29323, -3396, 17845, -9966},
         {-27884, 23786, -23003, -29878},
         {-24952, -2465, -12527, 9087}},
        {{31066, 19881, 14863, 16264},
         {17499, 19391, -23792, -25706},
         {16590, 11765, -10792, -12759}},
    };
    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int16x4_t a = vld1_s16(test_vec[i].a);
        int16x4_t b = vld1_s16(test_vec[i].b);
        int16x4_t r = vqrdmulh_s16(a, b);
        int16x4_t check = vld1_s16(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
