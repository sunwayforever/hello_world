// 2022-12-06 10:52
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_barrier() {
    printf("------ %s ------\n", __FUNCTION__);
    /* NOTE: block 之间有隐式的 barrier, 在 block 内部, 可以通过 omp barrier 插
     * 入一个 barrier */
#pragma omp parallel num_threads(2)
    {
        printf("%d\n", omp_get_thread_num());
#pragma omp barrier
        printf("%d\n", omp_get_thread_num());
    }
}

void test_nowait() {
    printf("------ %s ------\n", __FUNCTION__);
    /* NOTE: omp for 后面会有一个隐式的 barrier, 通过 omp nowait 可以去掉这个
     * barrier */
#pragma omp parallel num_threads(2)
    {
#pragma omp for nowait
        for (int i = 0; i < 2; i++) {
            printf("%d\n", omp_get_thread_num());
        }
        printf("---%d---\n", omp_get_thread_num());
    }
}

void test_master() {
    printf("------ %s ------\n", __FUNCTION__);
#pragma omp parallel num_threads(2)
    {
        /* NOTE: omp master 相当于 if (omp_get_thread_num==0) {xxx} */
#pragma omp master
        {
            printf("master %d\n", omp_get_thread_num());
            printf("master %d\n", omp_get_thread_num());
            printf("master %d\n", omp_get_thread_num());
        }
        printf("%d\n", omp_get_thread_num());
    }
}

void test_single() {
    printf("------ %s ------\n", __FUNCTION__);
#pragma omp parallel num_threads(2)
    {
        /* NOTE: single 与 master 类似, 相当于:
         * if (omp_get_thread_num==N) {xxx}; barrier();
         *
         * 1. single 不一定需要在 master thread
         * 2. single 后面有一个隐式的 barrier
         * */
#pragma omp single
        {
            printf("single %d\n", omp_get_thread_num());
            printf("single %d\n", omp_get_thread_num());
            printf("single %d\n", omp_get_thread_num());
        }
        printf("%d\n", omp_get_thread_num());
    }
}

void test_lock() {
    printf("------ %s ------\n", __FUNCTION__);
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel num_threads(2)
    {
        omp_set_lock(&lock);
        for (int i = 0; i < 5; i++) {
            printf("%d\n", omp_get_thread_num());
        }
        omp_unset_lock(&lock);
    }
}

void test_flush() {
    printf("------ %s ------\n", __FUNCTION__);
    int flag = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp master
        { flag = 1; }
        /* NOTE: 由于 omp master 没有隐式的 barrier, 所以其它线程需要用 omp
         * flush 获得 flag = 1 的改动 */
        while (flag != 1) {
#pragma omp flush(flag)
        }
        printf("%d %d\n", flag, omp_get_thread_num());
    }
}

int main(int argc, char *argv[]) {
    test_barrier();
    test_nowait();
    test_master();
    test_single();
    test_lock();
    test_flush();
}
