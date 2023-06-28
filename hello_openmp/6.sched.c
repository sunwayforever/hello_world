// 2022-12-06 11:42
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TIMEIT_REPS 3
#define TIMEIT(F)                                              \
    {                                                          \
        double start = omp_get_wtime();                        \
        for (int i = 0; i < TIMEIT_REPS; ++i) {                \
            F;                                                 \
        }                                                      \
        double diff = (omp_get_wtime() - start) / TIMEIT_REPS; \
        printf("%s : %f sec\n", #F, diff);                     \
    }

void test_section() {
    printf("--- %s ---\n", __FUNCTION__);
#pragma omp parallel num_threads(2)
    /* NOTE: omp sections 与 omp for 有一点类似:
     * 它负责对任务进行拆分, 但它不是针对 for, 而是针对几个独立的 section.
     * 后续的 3 个 section 会被分发给不同的 thread 执行.
     *
     * sections 与 openmp 3.0 新增的 task 很相似, 只是 task 不再需要外层的 omp
     * sections 这个 construct */
#pragma omp sections
    {
#pragma omp section
        printf("%d\n", omp_get_thread_num());
#pragma omp section
        printf("%d\n", omp_get_thread_num());
#pragma omp section
        printf("%d\n", omp_get_thread_num());
    }
    /* NOTE: omp sections 后有一个隐式的 barrier */
#pragma omp sections
    {
#pragma omp section
        printf("1 %d\n", omp_get_thread_num());
#pragma omp section
        printf("1 %d\n", omp_get_thread_num());
#pragma omp section
        printf("1 %d\n", omp_get_thread_num());
    }
}

void test_schedule() {
    printf("--- %s ---\n", __FUNCTION__);
    /* NOTE: schedule 是指 omp for 是如何进行 work sharing
     * 在 2.pi.c 中进行手动的 work sharing 时用了两种方法:
     *
     * 1. thread_n 处理 [n+k*NUM_THREADS for k in 0,1,2,...]
     * 2. thread_n 处理 [n*STEP..(n+1)*STEP], 其中 STEP=total/NUM_THREADS
     *
     * 前者相当于 schedule(static, 1), 后者是相当于 schedule(static, STEP)
     *
     * 通过调整 schedule, 可以使单个线程有较好的 locality */

#pragma omp parallel for schedule(static, 1) num_threads(2)
    for (int i = 0; i < 6; i++) {
        printf("%d %d\n", i, omp_get_thread_num());
    }
    printf("------\n");
#pragma omp parallel for schedule(static, 6 / 2) num_threads(2)
    for (int i = 0; i < 6; i++) {
        printf("%d %d\n", i, omp_get_thread_num());
    }
}

int data[1024][102400];
void test_schedule_locality_1() {
    for (int i = 0; i < 1024; i++) {
#pragma omp parallel for schedule(static, 1) num_threads(4)
        for (int j = 0; j < 102400; j++) {
            data[i][j] = 1;
        }
    }
}
void test_schedule_locality_32() {
    for (int i = 0; i < 1024; i++) {
#pragma omp parallel for schedule(static, 32) num_threads(4)
        for (int j = 0; j < 102400; j++) {
            data[i][j] = 1;
        }
    }
}

void test_schedule_dynamic_1() {
    /* NOTE: 有时 omp for 中每个循环的时间差别较大, 这时使用 static schedule 会
     * 导致某些线程比其它线程提前处理完而无事可做.
     *
     * 通过 dynamic schedule, 线程会以类似线程池的方式在运行时按需获得任务. 由于
     * dynamic schedule 有运行时开销, 所以有时会比 static schedule 更慢
     * */
    for (int i = 0; i < 1024; i++) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
        for (int j = 0; j < 102400; j++) {
            data[i][j] = 1;
        }
    }
}

void test_schedule_dynamic_32() {
    /* NOTE: 有时 omp for 中每个循环的时间差别较大, 这时使用 static schedule 会
     * 导致某些线程比其它线程提前处理完而无事可做.
     *
     * 通过 dynamic schedule, 线程会以类似线程池的方式在运行时按需获得任务. 由于
     * dynamic schedule 有运行时开销, 所以有时会比 static schedule 更慢
     * */
    for (int i = 0; i < 1024; i++) {
#pragma omp parallel for schedule(dynamic, 32) num_threads(4)
        for (int j = 0; j < 102400; j++) {
            data[i][j] = 1;
        }
    }
}

void test_schedule_perfmance() {
    printf("--- %s ---\n", __FUNCTION__);
    TIMEIT(test_schedule_locality_1());
    TIMEIT(test_schedule_locality_32());
    TIMEIT(test_schedule_dynamic_1());
    TIMEIT(test_schedule_dynamic_32());
}

int main(int argc, char *argv[]) {
    test_section();
    test_schedule();
    test_schedule_perfmance();
}
