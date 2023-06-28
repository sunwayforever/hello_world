// 2023-05-04 13:46
#ifndef UTIL_H
#define UTIL_H

#define COPY(a, b)                                      \
    do {                                                \
        int n = sizeof(a.values) / sizeof(a.values[0]); \
        for (int i = 0; i < n; i++) {                   \
            a.values[i] = b.values[i];                  \
        }                                               \
    } while (0)

#define COPY_HIGH(a, b)                                 \
    do {                                                \
        int n = sizeof(a.values) / sizeof(a.values[0]); \
        for (int i = 0; i < n; i++) {                   \
            a.values[i] = b.values[i + 8];              \
        }                                               \
    } while (0)

#define MERGE(c, a, b)                                  \
    do {                                                \
        int n = sizeof(a.values) / sizeof(a.values[0]); \
        for (int i = 0; i < n; i++) {                   \
            c.values[i] = a.values[i];                  \
            c.values[i + 8] = b.values[i];              \
        }                                               \
    } while (0)

#endif  // UTIL_H
