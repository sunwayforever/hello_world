#include <stdio.h>

int a = 1;
float b = 1.0;

int main(int argc, char* argv[]) {
    __set_a(20);
    printf("%d\n", __get_a());
    printf("%f\n", __get_b());
}
