// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* NOTE: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
 *
 * 关于 ring allreduce:
 *
 * ring allreduce 需要传递的总数据量与普通的算法是一样的, 例如普通的算法可以简单
 * 的循环 N 次, 每次把第 n 个节点的数据传递给最后一个节点. ring allreduce 的好处
 * 是它每次循环时可以充分利用设备之间的带宽, 而不像普通算法那样, 两个设备之间带
 * 宽需求很高, 其它设备的带宽空闲.
 *
 * 因此, 如果设备之间使用的是总线型或星型拓朴, 则 ring allreduce 就没有优势了
 * */
int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int number[12] = {0};
    if (pid == 0) {
        for (int i = 0; i < 12; i++) {
            number[i] = i;
        }
    }

    int sub_number[3] = {0};
    MPI_Scatter(number, 3, MPI_INT, sub_number, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < 3; i++) {
        local_sum += sub_number[i];
    }
    int sum = 0;
    MPI_Reduce(&local_sum, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (pid == 0) {
        printf("%d\n", sum);
    }

    MPI_Allreduce(&local_sum, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("pid: %d, %d\n", pid, sum);
    MPI_Finalize();
    return 0;
}
