#include <omp.h>
#include <stdio.h>
int main() {
    int x = 1;
#pragma omp target map(tofrom : x)
    for (int i = 0; i < 10; i++) {
        x += i;
    }
    printf("x = %d\n", x);
    return 0;
}
