// 2022-03-05 19:48
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern int foo();
extern int foo2();
int main(int argc, char *argv[]) { printf("%d %d\n", foo(), foo2()); }
