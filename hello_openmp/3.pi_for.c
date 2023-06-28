// 2022-12-05 15:26
/* https://www.openmp.org/wp-content/uploads/omp-hands-on-SC08.pdf */
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

double pi_parallel_for() {
    double step = 1.0 / N_STEPS;
    double sum = 0.0;
#pragma omp parallel
    {
        double x = 0.0;
        double partial_sum = 0.0;
#pragma omp for
        /* NOTE: omp for 可以自动做 work sharing, 但必须直接在 for (xxx) 之前 */
        for (int i = 0; i < N_STEPS; i++) {
            x = (i + 0.5) * step;
            partial_sum += 4.0 / (1.0 + x * x);
        }
#pragma omp atomic
        sum += partial_sum;
    }
    return step * sum;
}

double pi_parallel_for_reduction() {
    double step = 1.0 / N_STEPS;
    double partial_sum[NUM_THREADS] = {0.0};
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    /* NOTE: omp parallel for 相当于 parallem + for, 这里用到 reduction,
     * reduction (+:sum)相当于:
     *
     * 1. 声明了一个线程内部的 local_sum, 其初始值为 0 (+ 对应的初始值为 0, *
     * 的话会是 1)
     * 2. local_sum 在线程内部被 `plus reduce`
     * 3. 所有 local_sum 最后再通过一次 `plus reduce` 得到最终的 sum
     *
     * 除了 +, 还支持 *, -, &, |, ^, &&, ||
     *
     * 通过 parallel for 和 reduction, pi_linear 基本可以无修改的替换成 omp 版本
     *  */
    for (int i = 0; i < N_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
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
    TIMEIT(pi_parallel_for());
    TIMEIT(pi_parallel_for_reduction());
}
