// 2023-04-21 18:05
#include <neon.h>
#include <neon_test.h>
// int8x8_t vtbx2_s8(int8x8_t a,int8x8x2_t b,int8x8_t idx)
// uint8x8_t vtbx2_u8(uint8x8_t a,uint8x8x2_t b,uint8x8_t idx)
// poly8x8_t vtbx2_p8(poly8x8_t a,poly8x8x2_t b,uint8x8_t idx)
// int8x8_t vtbx3_s8(int8x8_t a,int8x8x3_t b,int8x8_t idx)
// uint8x8_t vtbx3_u8(uint8x8_t a,uint8x8x3_t b,uint8x8_t idx)
// poly8x8_t vtbx3_p8(poly8x8_t a,poly8x8x3_t b,uint8x8_t idx)
// int8x8_t vtbx4_s8(int8x8_t a,int8x8x4_t b,int8x8_t idx)
// uint8x8_t vtbx4_u8(uint8x8_t a,uint8x8x4_t b,uint8x8_t idx)
// poly8x8_t vtbx4_p8(poly8x8_t a,poly8x8x4_t b,uint8x8_t idx)
// int8x8_t vqtbx1_s8(int8x8_t a,int8x16_t t,uint8x8_t idx)
// int8x16_t vqtbx1q_s8(int8x16_t a,int8x16_t t,uint8x16_t idx)
// uint8x8_t vqtbx1_u8(uint8x8_t a,uint8x16_t t,uint8x8_t idx)
// uint8x16_t vqtbx1q_u8(uint8x16_t a,uint8x16_t t,uint8x16_t idx)
// poly8x8_t vqtbx1_p8(poly8x8_t a,poly8x16_t t,uint8x8_t idx)
// poly8x16_t vqtbx1q_p8(poly8x16_t a,poly8x16_t t,uint8x16_t idx)
// int8x8_t vqtbx2_s8(int8x8_t a,int8x16x2_t t,uint8x8_t idx)
// int8x16_t vqtbx2q_s8(int8x16_t a,int8x16x2_t t,uint8x16_t idx)
// uint8x8_t vqtbx2_u8(uint8x8_t a,uint8x16x2_t t,uint8x8_t idx)
// uint8x16_t vqtbx2q_u8(uint8x16_t a,uint8x16x2_t t,uint8x16_t idx)
// poly8x8_t vqtbx2_p8(poly8x8_t a,poly8x16x2_t t,uint8x8_t idx)
// poly8x16_t vqtbx2q_p8(poly8x16_t a,poly8x16x2_t t,uint8x16_t idx)
// int8x8_t vqtbx3_s8(int8x8_t a,int8x16x3_t t,uint8x8_t idx)
// int8x16_t vqtbx3q_s8(int8x16_t a,int8x16x3_t t,uint8x16_t idx)
// uint8x8_t vqtbx3_u8(uint8x8_t a,uint8x16x3_t t,uint8x8_t idx)
// uint8x16_t vqtbx3q_u8(uint8x16_t a,uint8x16x3_t t,uint8x16_t idx)
// poly8x8_t vqtbx3_p8(poly8x8_t a,poly8x16x3_t t,uint8x8_t idx)
// poly8x16_t vqtbx3q_p8(poly8x16_t a,poly8x16x3_t t,uint8x16_t idx)
// int8x8_t vqtbx4_s8(int8x8_t a,int8x16x4_t t,uint8x8_t idx)
// int8x16_t vqtbx4q_s8(int8x16_t a,int8x16x4_t t,uint8x16_t idx)
// uint8x8_t vqtbx4_u8(uint8x8_t a,uint8x16x4_t t,uint8x8_t idx)
// uint8x16_t vqtbx4q_u8(uint8x16_t a,uint8x16x4_t t,uint8x16_t idx)
// poly8x8_t vqtbx4_p8(poly8x8_t a,poly8x16x4_t t,uint8x8_t idx)
// poly8x16_t vqtbx4q_p8(poly8x16_t a,poly8x16x4_t t,uint8x16_t idx)

TEST_CASE(test_simde_vtbx2_s8) {
    struct {
        int8_t a[8];
        int8_t b[2][8];
        int8_t c[8];
        int8_t r[8];
    } test_vec[] = {
        {{43, -44, -14, -19, 84, 72, 125, 94},
         {{106, -98, 24, -9, -40, 36, -25, 3},
          {113, 40, -48, -2, -25, -61, 90, -92}},
         {13, 8, 0, 9, 7, 3, 5, 2},
         {-61, 113, 106, 40, 3, -9, 36, 24}},
        {{75, -61, -95, 35, -25, -120, 38, 89},
         {{-80, -9, 87, -105, -70, -79, 59, -57},
          {-103, -21, -111, -80, 15, -106, -14, 7}},
         {14, 33, 3, 14, -50, 14, 8, 25},
         {-14, -61, -105, -14, -25, -14, -103, 89}},
        {{90, 89, -22, 20, 11, 37, -37, -92},
         {{17, 108, 85, 32, 3, 71, 39, -111},
          {105, 122, 96, 55, 121, -40, 80, 58}},
         {2, 12, 3, 4, 15, 5, 6, 73},
         {85, 121, 32, 3, 58, 71, 39, -92}},
        {{-91, 67, 39, -88, -117, 78, 58, -12},
         {{-56, -102, 43, 65, 114, 123, 124, 100},
          {7, 95, -8, -9, 69, -33, 64, -124}},
         {15, 14, 14, 5, -41, 12, 12, 124},
         {-124, 64, 64, 123, -117, 69, 69, -12}},
        {{-7, -86, 43, 107, 37, -89, -48, 44},
         {{7, -56, 35, 76, -89, 100, -48, 87},
          {2, -98, -4, -39, 90, -87, 85, 90}},
         {124, 10, -27, 14, 9, 4, 8, 2},
         {-7, -4, 43, 85, -98, -89, 2, 35}},
        {{-42, -125, -82, 126, -25, 126, -43, -23},
         {{28, -47, -62, 118, 122, 24, -47, -9},
          {-110, -74, -107, 108, 27, 29, 62, 41}},
         {0, 12, 13, 11, -118, 13, 13, 1},
         {28, 27, 29, 108, -25, 29, 29, -47}},
        {{-123, -44, 76, 0, -20, 29, -9, 126},
         {{-45, -116, -22, -18, -87, 41, 24, 121},
          {-91, 117, -91, 48, 50, 50, -111, 114}},
         {110, 112, -102, 7, 4, 12, 13, 9},
         {-123, -44, 76, 121, -87, 50, 50, 117}},
        {{44, -43, 40, -43, -2, 64, 79, -93},
         {{-75, -12, -45, -25, 38, 100, 89, -108},
          {-44, -12, -68, -8, -96, -71, -94, 32}},
         {2, 11, 12, 8, -20, 6, 1, 8},
         {-45, -8, -96, -44, -2, 89, -12, -44}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8x2_t b;
        b.val[0] = vld1_s8(test_vec[i].b[0]);
        b.val[1] = vld1_s8(test_vec[i].b[1]);
        int8x8_t c = vld1_s8(test_vec[i].c);
        int8x8_t r = vtbx2_s8(a, b, c);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
