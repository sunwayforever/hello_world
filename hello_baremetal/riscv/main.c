#include <stdint.h>

/* NOTE: 为什么需要定义 tohost, fromhost
 * https://github.com/riscv-software-src/riscv-isa-sim/issues/364 */
int64_t tohost;
int64_t fromhost;
void main () {
    int x = 1;
    int y = x + 1;
    tohost = 1;
}
