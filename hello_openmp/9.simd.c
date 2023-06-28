// 2023-04-11 13:22
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int a[] = {1, 2, 3, 4, 5, 6};
    int b[] = {1, 2, 3, 4, 5, 6};
    int c[] = {1, 2, 3, 4, 5, 6};
#pragma omp simd
    for (int i = 0; i < 4; i++) {
        a[i] = a[i] + b[i] * c[i];
    }
    for (int i = 0; i < 4; i++) {
        printf("%d\n", a[i]);
    }
}
