#include <stdio.h>

void trace(char* func) { printf("---%s---\n", func); }
void foo() {}

int main(int argc, char* argv[]) {
    foo();
    return 0;
}
