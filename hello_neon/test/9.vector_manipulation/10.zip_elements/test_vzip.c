// 2023-04-21 15:45
#include <neon.h>
#include <neon_test.h>

TEST_CASE(test_vzip1_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{75, 64, 23, -70, -20, -62, -104, 0},
         {77, -69, -118, 40, -21, 81, 79, -43},
         {75, 77, 64, -69, 23, -118, -70, 40}},
        {{-10, 47, -91, -90, 57, -23, -71, -7},
         {-83, 112, 83, -15, 26, -11, -125, 101},
         {-10, -83, 47, 112, -91, 83, -90, -15}},
        {{53, -102, 32, 33, 93, -72, 33, -86},
         {115, -84, -46, 95, -3, 33, 52, -13},
         {53, 115, -102, -84, 32, -46, 33, 95}},
        {{81, -39, -103, -118, -62, 82, -125, 111},
         {-61, -42, 97, -35, -53, -28, 67, 1},
         {81, -61, -39, -42, -103, 97, -118, -35}},
        {{INT8_MAX, 99, 34, -36, 27, 68, -122, -114},
         {-16, 88, -19, -19, 122, 33, -31, -53},
         {INT8_MAX, -16, 99, 88, 34, -19, -36, -19}},
        {{-5, 122, 85, -67, -51, -40, 45, -112},
         {-82, -114, 109, 121, 114, -80, 122, -15},
         {-5, -82, 122, -114, 85, 109, -67, 121}},
        {{19, -99, -51, 46, -31, 83, -67, -47},
         {-84, -86, -66, 38, -52, -97, -15, -57},
         {19, -84, -99, -86, -51, -66, 46, 38}},
        {{26, 70, -124, -25, 30, -79, 119, -52},
         {63, -28, 69, -78, -107, -64, -93, -88},
         {26, 63, 70, -28, -124, 69, -25, -78}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vzip1_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vzip2_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[8];
    } test_vec[] = {
        {{70, -80, -52, 94, 33, 70, -2, 95},
         {-15, 85, INT8_MAX, 75, -126, 59, 49, -48},
         {33, -126, 70, 59, -2, 49, 95, -48}},
        {{95, 87, 16, 40, 102, -8, 74, -100},
         {-117, -49, 35, -122, -72, -81, -87, -2},
         {102, -72, -8, -81, 74, -87, -100, -2}},
        {{95, 117, 93, -127, -68, 91, -32, -83},
         {-80, 96, -8, 50, -101, 42, 3, -6},
         {-68, -101, 91, 42, -32, 3, -83, -6}},
        {{-127, 19, 34, -25, 11, 109, -125, -106},
         {60, -89, 28, -12, 86, -59, -13, -75},
         {11, 86, 109, -59, -125, -13, -106, -75}},
        {{59, 80, 54, -9, -85, 23, -92, 91},
         {119, -100, -114, 18, -58, -111, 12, 72},
         {-85, -58, 23, -111, -92, 12, 91, 72}},
        {{-92, 47, 47, -81, -100, -77, 69, -40},
         {90, 97, -51, -80, 38, -64, 101, 97},
         {-100, 38, -77, -64, 69, 101, -40, 97}},
        {{16, -100, 88, -69, -77, -4, 22, 42},
         {-103, -92, 60, 95, 53, 72, -89, -39},
         {-77, 53, -4, 72, 22, -89, 42, -39}},
        {{119, -41, -120, 19, -118, -51, -20, -28},
         {46, -71, -108, 85, 121, -7, -74, -119},
         {-118, 121, -51, -7, -20, -74, -28, -119}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8_t r = vzip2_s8(a, b);
        int8x8_t check = vld1_s8(test_vec[i].r);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}

TEST_CASE(test_vzip_s8) {
    struct {
        int8_t a[8];
        int8_t b[8];
        int8_t r[2][8];
    } test_vec[] = {
        {
            {-26, -14, -73, -66, 84, -37, -44, 48},
            {-17, 122, -84, -31, -118, 66, -5, -66},
            {
                {-26, -17, -14, 122, -73, -84, -66, -31},
                {84, -118, -37, 66, -44, -5, 48, -66},
            },
        },
        {
            {-65, -32, 65, -83, 64, 26, -27, -82},
            {10, -11, -8, 126, INT8_MAX, 56, -15, 101},
            {
                {-65, 10, -32, -11, 65, -8, -83, 126},
                {64, INT8_MAX, 26, 56, -27, -15, -82, 101},
            },
        },
        {
            {43, -87, 35, INT8_MAX, -124, -9, -80, 115},
            {114, 92, 85, -4, -98, 80, -70, 94},
            {
                {43, 114, -87, 92, 35, 85, INT8_MAX, -4},
                {-124, -98, -9, 80, -80, -70, 115, 94},
            },
        },
        {
            {48, -5, 11, 113, 21, -16, 31, 32},
            {-27, 24, -98, 100, 80, -112, -55, 123},
            {
                {48, -27, -5, 24, 11, -98, 113, 100},
                {21, 80, -16, -112, 31, -55, 32, 123},
            },
        },
        {
            {57, -19, -5, -67, -28, -85, 49, 86},
            {7, -122, 82, -90, -42, 13, 4, 6},
            {
                {57, 7, -19, -122, -5, 82, -67, -90},
                {-28, -42, -85, 13, 49, 4, 86, 6},
            },
        },
        {
            {8, 15, 119, 30, -1, -105, 62, -28},
            {-81, -36, 72, -1, 108, 17, 123, -91},
            {
                {8, -81, 15, -36, 119, 72, 30, -1},
                {-1, 108, -105, 17, 62, 123, -28, -91},
            },
        },
        {
            {-2, 118, 99, -29, 33, -108, 57, 40},
            {26, -116, -50, -16, -103, -46, -10, -95},
            {
                {-2, 26, 118, -116, 99, -50, -29, -16},
                {33, -103, -108, -46, 57, -10, 40, -95},
            },
        },
        {
            {-31, 110, -65, -32, 5, -3, -60, -76},
            {-38, 12, -77, 70, 30, 46, -20, 28},
            {
                {-31, -38, 110, 12, -65, -77, -32, 70},
                {5, 30, -3, 46, -60, -20, -76, 28},
            },
        },

    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vld1_s8(test_vec[i].a);
        int8x8_t b = vld1_s8(test_vec[i].b);
        int8x8x2_t r = vzip_s8(a, b);

        int8x8_t check1 = vld1_s8(test_vec[i].r[0]);
        int8x8_t check2 = vld1_s8(test_vec[i].r[1]);
        ASSERT_EQUAL(r.val[0], check1);
        ASSERT_EQUAL(r.val[1], check2);
    }
    return 0;
}
