#include <stdio.h>

void trace(char* func) { printf("%s\n", func); }

int foo() { return 0; }

int main(int argc, char* argv[]) {
    foo();
    return 0;
}
