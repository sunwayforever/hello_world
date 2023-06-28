// 2022-12-05 15:26
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_THREADS 8
#define N_STEPS 100000000

#define TIMEIT_REPS 5
#define TIMEIT(F)                                              \
    {                                                          \
        double start = omp_get_wtime();                        \
        for (int i = 0; i < TIMEIT_REPS; ++i) {                \
            F;                                                 \
        }                                                      \
        double diff = (omp_get_wtime() - start) / TIMEIT_REPS; \
        printf("%f %50s : %f sec\n", F, #F, diff);             \
    }

double pi() {
    double step = 1.0 / N_STEPS;
    double partial_sum[NUM_THREADS] = {0.0};
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        double x = 0.0;
        double sum = 0.0;
        for (int i = id; i < N_STEPS; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        partial_sum[id] = sum;
    }
    double pi, sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        sum += partial_sum[i];
    }
    pi = step * sum;
    return pi;
}

double pi_another_work_sharing() {
    double step = 1.0 / N_STEPS;
    double partial_sum[NUM_THREADS] = {0.0};
    int STEP = (int)(N_STEPS / NUM_THREADS);
#pragma omp parallel
    {
        double x = 0.0;
        double sum = 0.0;
        int id = omp_get_thread_num();
        int start = id * STEP;
        int end = id * STEP + STEP;
        for (int i = start; i < end; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        partial_sum[id] = sum;
    }
    double pi, sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        sum += partial_sum[i];
    }
    pi = step * sum;
    return pi;
}

double pi_false_sharing() {
    double step = 1.0 / N_STEPS;
    /* NOTE: partial_sum 在同一个 cache line, 导致任何一个线程更新 partial_sum
     * 后都需要被同步到其它线程, 这种问题称为 false sharing
     *
     * http://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/multithreading_problems.html
     * */
    double partial_sum[NUM_THREADS] = {0.0};
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        double x = 0.0;
        for (int i = id; i < N_STEPS; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            partial_sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    double pi, sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        sum += partial_sum[i];
    }
    pi = step * sum;
    return pi;
}

double pi_no_false_sharing_with_align() {
    double step = 1.0 / N_STEPS;
    double* partial_sum[NUM_THREADS];
    /* NOTE: 强制 partial_sum[i] 在不同的 cache line, 假设 cache line <= 128 */
    for (int i = 0; i < NUM_THREADS; i++) {
        partial_sum[i] = (double*)malloc(128);
    }
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        double x = 0.0;
        for (int i = id; i < N_STEPS; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            *partial_sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    double pi, sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        sum += *partial_sum[i];
    }
    return step * sum;
}

double pi_no_false_sharing_with_atomic() {
    double step = 1.0 / N_STEPS;
    double sum = 0.0;
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        double x = 0.0;
        double partial_sum = 0.0;
        for (int i = id; i < N_STEPS; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            partial_sum += 4.0 / (1.0 + x * x);
        }
/* NOTE: critcal 要求后续操作是 exclusive 的, atomic 只要求对 sum 的更新是
 * exclusive 的 */
/* #pragma omp critical*/
#pragma omp atomic
        sum += partial_sum;
    }
    return step * sum;
}

double pi_linear() {
    double step = 1.0 / N_STEPS;
    double sum = 0.0;
    for (int i = 0; i < N_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(NUM_THREADS);
    TIMEIT(pi_linear());
    TIMEIT(pi_false_sharing());
    TIMEIT(pi_no_false_sharing_with_align());
    TIMEIT(pi_no_false_sharing_with_atomic());
    TIMEIT(pi());
    TIMEIT(pi_another_work_sharing());
}
