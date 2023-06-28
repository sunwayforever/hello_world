// 2023-04-18 14:17
#ifndef NEON_ASSERT_H
#define NEON_ASSERT_H

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#define _ASSERT_EQUAL(a, b, v)                \
    do {                                      \
        int n = sizeof(a v) / sizeof(a v[0]); \
        for (int i = 0; i < n; i++) {         \
            assert(a v[i] == b v[i]);         \
        }                                     \
    } while (0)

#define _ASSERT_CLOSE(a, b, v)                            \
    do {                                                  \
        int n = sizeof(a v) / sizeof(a v[0]);             \
        for (int i = 0; i < n; i++) {                     \
            if (isnanf(a v[i]) || isnanf(b v[i])) {       \
                assert(isnanf(b v[i]) && isnanf(a v[i])); \
            } else {                                      \
                assert(fabs(a v[i] - b v[i]) < 1e-2);     \
            }                                             \
        }                                                 \
    } while (0)

#define ASSERT_EQUAL_SCALAR(a, b) \
    do {                          \
        assert(a == b);           \
    } while (0)

#define ASSERT_CLOSE_SCALAR(a, b)   \
    do {                            \
        assert(fabs(a - b) < 1e-2); \
    } while (0)

#define TEST_CASE(name) __attribute__((constructor)) int name()

#ifdef __aarch64__
#define ASSERT_EQUAL(a, b) _ASSERT_EQUAL(a, b, )
#define ASSERT_CLOSE(a, b) _ASSERT_CLOSE(a, b, )
#else
#define ASSERT_EQUAL(a, b) _ASSERT_EQUAL(a, b, .values)
#define ASSERT_CLOSE(a, b) _ASSERT_CLOSE(a, b, .values)
#endif

int main(int argc, char *argv[]) { return 0; }

#endif  // NEON_ASSERT_H
