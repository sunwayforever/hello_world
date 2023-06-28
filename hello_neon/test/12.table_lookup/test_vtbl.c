// 2023-04-21 17:25
#include <neon.h>
#include <neon_test.h>
// int8x8_t vtbl1_s8(int8x8_t a,int8x8_t idx)
// uint8x8_t vtbl1_u8(uint8x8_t a,uint8x8_t idx)
// poly8x8_t vtbl1_p8(poly8x8_t a,uint8x8_t idx)
//
// int8x8_t vtbx1_s8(int8x8_t a,int8x8_t b,int8x8_t idx)
// uint8x8_t vtbx1_u8(uint8x8_t a,uint8x8_t b,uint8x8_t idx)
// poly8x8_t vtbx1_p8(poly8x8_t a,poly8x8_t b,uint8x8_t idx)
//
// int8x8_t vtbl2_s8(int8x8x2_t a,int8x8_t idx)
// uint8x8_t vtbl2_u8(uint8x8x2_t a,uint8x8_t idx)
// poly8x8_t vtbl2_p8(poly8x8x2_t a,uint8x8_t idx)
//
// int8x8_t vtbl3_s8(int8x8x3_t a,int8x8_t idx)
// uint8x8_t vtbl3_u8(uint8x8x3_t a,uint8x8_t idx)
// poly8x8_t vtbl3_p8(poly8x8x3_t a,uint8x8_t idx)
//
// int8x8_t vtbl4_s8(int8x8x4_t a,int8x8_t idx)
// uint8x8_t vtbl4_u8(uint8x8x4_t a,uint8x8_t idx)
// poly8x8_t vtbl4_p8(poly8x8x4_t a,uint8x8_t idx)
//
// int8x8_t vqtbl1_s8(int8x16_t t,uint8x8_t idx)
// int8x16_t vqtbl1q_s8(int8x16_t t,uint8x16_t idx)
// uint8x8_t vqtbl1_u8(uint8x16_t t,uint8x8_t idx)
// uint8x16_t vqtbl1q_u8(uint8x16_t t,uint8x16_t idx)
// poly8x8_t vqtbl1_p8(poly8x16_t t,uint8x8_t idx)
// poly8x16_t vqtbl1q_p8(poly8x16_t t,uint8x16_t idx)
//
// int8x8_t vqtbl2_s8(int8x16x2_t t,uint8x8_t idx)
// int8x16_t vqtbl2q_s8(int8x16x2_t t,uint8x16_t idx)
// uint8x8_t vqtbl2_u8(uint8x16x2_t t,uint8x8_t idx)
// uint8x16_t vqtbl2q_u8(uint8x16x2_t t,uint8x16_t idx)
// poly8x8_t vqtbl2_p8(poly8x16x2_t t,uint8x8_t idx)
// poly8x16_t vqtbl2q_p8(poly8x16x2_t t,uint8x16_t idx)
//
// int8x8_t vqtbl3_s8(int8x16x3_t t,uint8x8_t idx)
// int8x16_t vqtbl3q_s8(int8x16x3_t t,uint8x16_t idx)
// uint8x8_t vqtbl3_u8(uint8x16x3_t t,uint8x8_t idx)
// uint8x16_t vqtbl3q_u8(uint8x16x3_t t,uint8x16_t idx)
// poly8x8_t vqtbl3_p8(poly8x16x3_t t,uint8x8_t idx)
// poly8x16_t vqtbl3q_p8(poly8x16x3_t t,uint8x16_t idx)
//
// int8x8_t vqtbl4_s8(int8x16x4_t t,uint8x8_t idx)
// int8x16_t vqtbl4q_s8(int8x16x4_t t,uint8x16_t idx)
// uint8x8_t vqtbl4_u8(uint8x16x4_t t,uint8x8_t idx)
// uint8x16_t vqtbl4q_u8(uint8x16x4_t t,uint8x16_t idx)
// poly8x8_t vqtbl4_p8(poly8x16x4_t t,uint8x8_t idx)
// poly8x16_t vqtbl4q_p8(poly8x16x4_t t,uint8x16_t idx)

TEST_CASE(test_simde_vtbl1_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{99, -37, -95, INT8_MAX, -56, -46, -47, -86},
         {2, -89, 7, 1, 116, 7, 5, 3},
         {-95, 0, -86, -37, 0, -86, -46, INT8_MAX}},
        {{121, -56, -37, -111, -103, -111, 17, -3},
         {4, 2, 4, 4, 5, 5, 6, 7},
         {-103, -37, -103, -103, -111, -111, 17, -3}},
        {{-81, -56, -30, 116, 86, -72, -31, -48},
         {INT8_MIN, 4, 1, 2, 5, 2, 7, 1},
         {0, 86, -56, -30, -72, -30, -48, -56}},
        {{-127, 17, 63, 79, INT8_MAX, 35, 42, 46},
         {3, 5, 3, 1, -59, 4, 1, 70},
         {79, 35, 79, 17, 0, INT8_MAX, 17, 0}},
        {{10, 51, -76, -21, -2, -11, -63, INT8_MIN},
         {7, 0, -49, 6, 3, 1, 5, 6},
         {INT8_MIN, 10, 0, -63, -21, 51, -11, -63}},
        {{-44, 114, -87, -70, -23, -17, -60, -13},
         {2, 1, -33, 0, 6, 0, 0, 5},
         {-87, 114, 0, -44, -60, -44, -44, -17}},
        {{9, 35, 59, -27, -124, 77, 1, 89},
         {7, 2, 3, 0, 1, 7, 4, 3},
         {89, 59, -27, 9, 35, 89, -124, -27}},
        {{-21, 48, -127, 84, -31, 84, -60, -22},
         {7, 7, 7, 3, 5, 0, 4, 4},
         {-22, -22, -22, 84, 84, -21, -31, -31}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vtbl1_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vtbl2_s8) {
    struct {
        int8_t a[2][8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{{-96, 73, -46, -52, 62, 65, 82, -124},
          {-84, 54, 67, 104, -32, -89, 38, -123}},
         {8, 0, 9, -5, 110, 5, 104, 88},
         {-84, -96, 54, 0, 0, 65, 0, 0}},
        {{{-28, 46, -69, 34, 111, 13, -89, 27},
          {67, -22, -124, 35, -110, -86, -87, 106}},
         {10, 2, 6, 9, 7, 14, 1, 3},
         {-124, -69, -89, -22, 27, -87, 46, 34}},
        {{{-98, 79, 126, 14, 92, 37, 41, -97},
          {15, -83, -62, -95, 87, 107, 12, -46}},
         {13, 2, 11, 5, 0, -4, 8, 6},
         {107, 126, -95, 37, -98, 0, 15, 41}},
        {{{30, 42, 25, 122, 79, 67, 25, 95},
          {-16, -36, 0, 72, 71, 12, 26, -11}},
         {14, 5, 10, -65, 1, 2, 85, 11},
         {26, 67, 0, 0, 42, 25, 0, 72}},
        {{{-45, INT8_MIN, -7, 34, -61, 19, -127, -76},
          {-17, -126, -4, 54, -114, 22, 43, 13}},
         {11, 5, 12, 12, 7, 1, 103, 6},
         {54, 19, -114, -114, -76, INT8_MIN, 0, -127}},
        {{{89, -117, 1, 28, -98, -125, -48, -115},
          {5, -52, -60, -109, -30, -17, -96, -51}},
         {5, 12, 10, 12, 14, 1, 98, 0},
         {-125, -30, -60, -30, -96, -117, 0, 89}},
        {{{113, 65, 34, 15, -60, -14, -99, -55},
          {-65, 97, 93, -95, 80, -3, 111, 117}},
         {106, 9, 2, -8, 10, 4, 8, 15},
         {0, 97, 34, 0, 93, -60, -65, 117}},
        {{{34, -52, -14, -26, -65, -113, -80, 126},
          {-16, 13, 31, 64, 10, -114, -74, 116}},
         {7, 8, 12, 2, -100, 5, -79, -107},
         {126, -16, 10, -14, 0, -113, 0, 0}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8x2_t a;
        a.val[0] = vld1_s8(test_vec[i].a[0]);
        a.val[1] = vld1_s8(test_vec[i].a[1]);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vtbl2_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vtbx1_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t c[8];
        int8_t r[8];
    } test_vec[] = {
        {{37, -53, 27, 42, -10, -65, 122, -112},
         {-51, -51, 80, 6, -1, -100, -123, 47},
         {1, 112, 7, 3, -80, -60, 3, -89},
         {-51, -53, 47, 6, -10, -65, 6, -112}},
        {{123, 67, -58, 113, 2, 64, 1, -48},
         {13, 81, -42, 12, -18, 91, 60, -17},
         {3, 3, 2, 4, 0, 5, 3, 7},
         {12, 12, -42, -18, 13, 91, 12, -17}},
        {{-59, -85, 93, -57, -21, 94, -105, -7},
         {-80, 109, 5, -98, -55, 65, -115, -108},
         {5, 0, 0, 5, 5, 3, 4, 1},
         {65, -80, -80, 65, 65, -98, -55, 109}},
        {{42, -79, 11, 22, 15, -93, 15, -65},
         {16, 20, 93, -39, 86, -21, 110, -101},
         {19, 6, 0, -112, 2, 4, 1, -89},
         {42, 110, 16, 22, 93, 86, 20, -65}},
        {{19, 24, 60, 34, -69, 75, -30, -53},
         {95, 63, -91, -75, 42, 19, 80, 61},
         {1, -8, 6, 3, 4, 7, 3, 6},
         {63, 24, 80, -75, 42, 61, -75, 80}},
        {{123, -119, -111, 54, -44, 115, 1, 51},
         {-78, -90, -23, -35, -71, 57, 26, 75},
         {50, 0, 6, 6, -104, 1, 4, -61},
         {123, -78, 26, 26, -44, -90, -71, 51}},
        {{-64, -50, 111, -108, 65, 112, -56, -13},
         {23, -79, -48, -48, -22, -21, 27, 28},
         {3, 2, -109, 3, 35, 7, 7, 6},
         {-48, -48, 111, -48, 65, 28, 28, 27}},
        {{112, 88, -90, -79, -55, 110, -92, -32},
         {31, 117, -80, 9, 96, -52, 38, 51},
         {118, 1, 7, -103, 80, 6, 0, 3},
         {112, 117, 51, -79, -55, 38, 31, 9}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t c = vld1_s8(test_vec[i].c);
        int8x8_t r = vtbx1_s8(a, b, c);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
