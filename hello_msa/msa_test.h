// 2023-04-18 14:17
#ifndef NEON_ASSERT_H
#define NEON_ASSERT_H

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef EMU
#define DEBUG_I(a)                                           \
    do {                                                     \
        printf("{");                                         \
        for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++) { \
            printf("%d, ", a[i]);                            \
        }                                                    \
        printf("}\n");                                       \
    } while (0)

#define DEBUG_F(a)                                           \
    do {                                                     \
        printf("{");                                         \
        for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++) { \
            printf("%f, ", a[i]);                            \
        }                                                    \
        printf("}\n");                                       \
    } while (0)

#define ASSERT_EQUAL(a, b)                \
    do {                                  \
        int n = sizeof(a) / sizeof(a[0]); \
        for (int i = 0; i < n; i++) {     \
            assert(a[i] == b[i]);         \
        }                                 \
    } while (0)

#define ASSERT_CLOSE(a, b)                            \
    do {                                              \
        int n = sizeof(a) / sizeof(a[0]);             \
        for (int i = 0; i < n; i++) {                 \
            if (isnanf(a[i]) || isnanf(b[i])) {       \
                assert(isnanf(b[i]) && isnanf(a[i])); \
            } else {                                  \
                assert(fabs(a[i] - b[i]) < 1e-2);     \
            }                                         \
        }                                             \
    } while (0)
#else
#define DEBUG_I(a)                                                  \
    do {                                                            \
        printf("{");                                                \
        for (int i = 0; i < sizeof(a) / sizeof(a.values[0]); i++) { \
            printf("%d, ", a.values[i]);                            \
        }                                                           \
        printf("}\n");                                              \
    } while (0)

#define DEBUG_F(a)                                                  \
    do {                                                            \
        printf("{");                                                \
        for (int i = 0; i < sizeof(a) / sizeof(a.values[0]); i++) { \
            printf("%f, ", a.values[i]);                            \
        }                                                           \
        printf("}\n");                                              \
    } while (0)

#define ASSERT_EQUAL(a, b)                       \
    do {                                         \
        int n = sizeof(a) / sizeof(a.values[0]); \
        for (int i = 0; i < n; i++) {            \
            assert(a.values[i] == b.values[i]);  \
        }                                        \
    } while (0)

#define ASSERT_CLOSE(a, b)                                          \
    do {                                                            \
        int n = sizeof(a) / sizeof(a.values[0]);                    \
        for (int i = 0; i < n; i++) {                               \
            if (isnanf(a.values[i]) || isnanf(b.values[i])) {       \
                assert(isnanf(b.values[i]) && isnanf(a.values[i])); \
            } else {                                                \
                assert(fabs(a.values[i] - b.values[i]) < 1e-2);     \
            }                                                       \
        }                                                           \
    } while (0)
#endif

#define ASSERT_EQUAL_SCALAR(a, b) \
    do {                          \
        assert(a == b);           \
    } while (0)

#define ASSERT_CLOSE_SCALAR(a, b)   \
    do {                            \
        assert(fabs(a - b) < 1e-2); \
    } while (0)

#define TEST_CASE(name) __attribute__((constructor)) int name()

int main(int argc, char *argv[]) { return 0; }

#endif  // NEON_ASSERT_H
