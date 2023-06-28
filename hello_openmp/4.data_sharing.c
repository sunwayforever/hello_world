// 2022-12-05 19:08
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 6

void test_private() {
    printf("---%s--- \n", __FUNCTION__);
    printf("------ shared -------\n");
    /* NOTE: 默认情况下 block 外的变量例如 sum 是 shared */
    int sum = 10;
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < N; i++) {
        /* NOTE: 这里缺少同步, sum += 1 可能会出现 race */
        sum += 1;
        printf("%d %d\n", sum, omp_get_thread_num());
    }
    printf("------ private -------\n");
    sum = 10;
    /* NOTE: 通过 private 让 block 获得一个 sum 的 private copy, 但这个 private
     * 的 sum 是未初始化的状态 */
#pragma omp parallel for private(sum) num_threads(2)
    for (int i = 0; i < N; i++) {
        sum += 1;
        printf("%d %d\n", sum, omp_get_thread_num());
    }
    printf("------ firstprivate -------\n");
    sum = 10;
    /* NOTE: 通过 firstprivate 让 block 获得一个 sum 的 private copy, 但 sum
     * 会初 始化为原始 sum 的值 1 */
#pragma omp parallel for firstprivate(sum) num_threads(2)
    for (int i = 0; i < N; i++) {
        sum += 1;
        printf("%d %d\n", sum, omp_get_thread_num());
    }
    printf("------ private -------\n");
    sum = 10;
#pragma omp parallel num_threads(2)
    {
        /* NOTE: 默认情况下 block 中的变量例如 private_sum 是 private */
        int private_sum = sum;
#pragma omp for
        for (int i = 0; i < N; i++) {
            private_sum += 1;
            printf("%d %d\n", private_sum, omp_get_thread_num());
        }
    }
}

void test_thread_private() {
    printf("---%s--- \n", __FUNCTION__);
    static int sum = 0;
#pragma omp threadprivate(sum)
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < N; i++) {
        sum += 1;
        printf("%d %d\n", sum, omp_get_thread_num());
    }
    printf("------ another block ------\n");
/* NOTE: threadprivate 是通过 TLS 实现的, 所以第二个 block 会复用前面的 sum 值.
 * 但这也导致它有额外的要求:
 *
 * 1. sum 变量需要有全局作用域
 * 2. 使用 threadprivate 时要求两个 block 使用相同的 thread num
 * */
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < N; i++) {
        sum += 1;
        printf("%d %d\n", sum, omp_get_thread_num());
    }
}

int main(int argc, char *argv[]) {
    test_private();
    test_thread_private();
}
