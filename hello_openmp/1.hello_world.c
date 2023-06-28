// 2022-12-05 14:51
#include <omp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    printf("------\n");
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("hello world %d\n", id);
    }

    printf("------\n");
#pragma omp parallel num_threads(4)
    {
        int id = omp_get_thread_num();
        printf("hello world %d\n", id);
    }

    printf("------\n");
    omp_set_num_threads(4);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("hello world %d\n", id);
    }
}
