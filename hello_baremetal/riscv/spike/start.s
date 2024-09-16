    .section .text
    .globl _start
_start:
    la sp, _end
    call main
