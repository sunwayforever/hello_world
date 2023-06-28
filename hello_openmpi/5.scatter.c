// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
    /* NOTE: number 的不同部分被分发给不同的 pid */
    int sub_number[3] = {0};
    MPI_Scatter(number, 3, MPI_INT, sub_number, 3, MPI_INT, 0, MPI_COMM_WORLD);

    printf("pid: %d\n", pid);
    for (int i = 0; i < 3; i++) {
        printf("%d ", sub_number[i]);
        sub_number[i] *= 2;
    }
    printf("\n");

    /* NOTE: 不同 pid 的 sub_number 被 gather 到 pid 0 对应的 number 中 */
    MPI_Gather(&sub_number, 3, MPI_INT, number, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("gather: pid: %d\n", pid);
        for (int i = 0; i < 12; i++) {
            printf("%d ", number[i]);
        }
        printf("\n");
    }

    /* NOTE: allgather 把 gather 结果分发给所有 pid, 而不仅仅是某一个 root */
    MPI_Allgather(&sub_number, 3, MPI_INT, number, 3, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("allgather: pid: %d\n", pid);
    for (int i = 0; i < 12; i++) {
        printf("%d ", number[i]);
    }
    printf("\n");
    MPI_Finalize();
    return 0;
}
