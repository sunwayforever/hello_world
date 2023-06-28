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
    int sub_number[3] = {0};
    MPI_Scatter(number, 3, MPI_INT, sub_number, 3, MPI_INT, 0, MPI_COMM_WORLD);

    float avg = 0.0f;
    for (int i = 0; i < 3; i++) {
        avg += (float)sub_number[i];
    }
    avg /= 3.0;

    float avgs[4] = {0.0f};
    MPI_Gather(&avg, 1, MPI_FLOAT, avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        float avg = 0.0;
        for (int i = 0; i < 4; i++) {
            avg += avgs[i];
        }
        printf("avg: %f\n", avg / 4.0);
    }
    MPI_Finalize();
    return 0;
}
