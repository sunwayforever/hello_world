// 2023-03-27 14:24
#include <stdio.h>
void bar(){};
__attribute__((notrace)) void foo() { bar(); };
void trace(char *func) { printf("---%s---\n", func); }
int main(int argc, char *argv[]) {
    foo();
    return 0;
}
