    .section .text
    .global _start_el1

_start_el1:
    mrs x9,CurrentEL
    asr x9, x9, 2
    ldr x10, =vector_table_el1
    msr vbar_el1, x10
    ldr x11, =_start_el0
    msr elr_el1, x11
    eret

_start_el0:
    svc #1

    .balign 0x800
vector_table_el1:
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
    .balign 0x80
    b _start_el1
