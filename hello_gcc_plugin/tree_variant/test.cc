#include <stdio.h>

int a = 1;

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; i++) {
        a += 1;
    }
    return a;
}
