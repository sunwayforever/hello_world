// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int number[10] = {0};
    if (pid == 0) {
        for (int i = 0; i < 10; i++) {
            number[i] = i;
        }
    }
    /* NOTE: number is broadcasted by 0, and recved by ALL pids, 这里不需要通过
     * MPI_Recv 来接收 broadcast, MPI_Bcast 既可以发送也可以接收 */
    MPI_Bcast(number, 10, MPI_INT, 0, MPI_COMM_WORLD);

    printf("pid: %d\n", pid);
    for (int i = 0; i < 10; i++) {
        printf("%d ", number[i]);
    }
    printf("\n");
    MPI_Finalize();
    return 0;
}
