// 2022-12-06 14:11
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TIMEIT(F)                                \
    {                                            \
        double start = omp_get_wtime();          \
        F;                                       \
        double diff = (omp_get_wtime() - start); \
        printf("%s : %f sec\n", #F, diff);       \
    }

void test_task() {
    printf("--- %s ---\n", __FUNCTION__);
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            /* NOTE: omp task 与 thread pool 非常相似, 与 omp sections 也类似,它
             * 主要是用在无法应用 omp for 的场景, 例如递归 */
#pragma omp task
            printf("%d\n", omp_get_thread_num());
#pragma omp task
            printf("%d\n", omp_get_thread_num());
#pragma omp task
            printf("%d\n", omp_get_thread_num());
        }
    }
}

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void swap(int *data, int a, int b) {
    int tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
}

void quick_sort_linear(int *data, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int pivort = lo;
    int curr = lo + 1;
    while (curr < hi) {
        if (data[curr] >= data[pivort]) {
            curr += 1;
        } else {
            swap(data, curr, pivort + 1);
            swap(data, pivort, pivort + 1);
            curr += 1;
            pivort += 1;
        }
    }
    quick_sort_linear(data, lo, pivort);
    quick_sort_linear(data, pivort + 1, hi);
}

void quick_sort_omp(int *data, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int pivort = lo;
    int curr = lo + 1;
    while (curr < hi) {
        if (data[curr] >= data[pivort]) {
            curr += 1;
        } else {
            swap(data, curr, pivort + 1);
            swap(data, pivort, pivort + 1);
            curr += 1;
            pivort += 1;
        }
    }

#pragma omp task
    quick_sort_omp(data, lo, pivort);
#pragma omp task
    quick_sort_omp(data, pivort + 1, hi);
}

void _quick_sort_omp(int *data, int lo, int hi) {
#pragma omp parallel num_threads(8)
    {
#pragma omp single
        quick_sort_omp(data, lo, hi);
    }
}

int data[1024 * 10000];
void test_quick_sort() {
    printf("--- %s ---\n", __FUNCTION__);
    int N = sizeof(data) / sizeof(int);
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }
    shuffle(data, N);
    TIMEIT(quick_sort_linear(data, 0, N));
    shuffle(data, N);
    TIMEIT(_quick_sort_omp(data, 0, N));
}

int main(int argc, char *argv[]) {
    test_task();
    test_quick_sort();
}
