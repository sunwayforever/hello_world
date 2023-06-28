// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int number = 0;
    if (pid == 0) {
        number = 10;
        printf("pid 0 send number %d to pid 1,2,3\n", number);
        /* NOTE: send 1 number of int to pid 1 */
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&number, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(&number, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
    } else {
        /* NOTE: receive 1 number of int from 0 */
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("pid %d received number %d from pid 0\n", pid, number);
    }
    MPI_Finalize();
    return 0;
}
