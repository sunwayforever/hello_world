// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    printf("pid: %d, np: %d\n", pid, np);
    MPI_Comm comm;
    /* NOTE: communator 把任务分成多个组, 相关 api 例如 MPI_Send, MPI_Bcast 等只
     * 能在组内部通信 */
    MPI_Comm_split(MPI_COMM_WORLD, pid % 2, pid, &comm);

    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &np);
    printf("pid: %d, np: %d\n", pid, np);

    int number = 0;
    if (pid == 0) {
        number = 10;
        MPI_Send(&number, 1, MPI_INT, 1, 0, comm);
    } else {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        printf("pid %d received number %d from pid 0\n", pid, number);
    }

    MPI_Finalize();
    return 0;
}
