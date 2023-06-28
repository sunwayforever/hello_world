// 2020-09-29 11:56
#include <stdlib.h>
volatile unsigned int *const UART0DR = (unsigned int *)0x101f1000;

void print_uart0(const char *s) {
    while (*s != '\0') {               /* Loop until end of string */
        *UART0DR = (unsigned int)(*s); /* Transmit char */
        s++;                           /* Next char */
    }
}

void main() {
    char *buff = (char*)malloc(10);
    /*
     * memset(buff, 0, 10);
     */
    print_uart0("Hello world!\n");
    /*
     * sprintf(buff, "hello");
     */
    /*
     * print_uart0(buff);
     */
}
