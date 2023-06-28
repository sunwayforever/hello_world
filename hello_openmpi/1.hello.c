// 2022-12-27 14:48
#include <mpi.h>
#include <stdio.h>
#include <string.h>

/* https://mpitutorial.com/tutorials/ */

int main(int argc, char *argv[]) {
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    printf("pid: %d, np: %d\n", pid, np);
    MPI_Finalize();
    return 0;
}
