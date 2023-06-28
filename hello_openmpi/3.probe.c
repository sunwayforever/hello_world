// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (pid == 0) {
        int number[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        printf("pid 0 send to pid 1,2,3\n");
        /* NOTE: send 1 number of int to pid 1 */
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&number, 5, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(&number, 10, MPI_INT, 3, 0, MPI_COMM_WORLD);
    } else {
        int n = 0;
        MPI_Status status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &n);
        int *number = (int *)malloc(n * sizeof(int));
        MPI_Recv(number, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("pid %d received %d number from pid 0\n", pid, n);
        for (int i = 0; i < n; i++) {
            printf("%d ", number[i]);
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
