ENTRY(_start_el1)
SECTIONS
{
 . = 0x40000000;
 .text : { *(.text) }
 .data : { *(.data) }
 __bss_start__ = .;
 .bss : { *(.bss COMMON) }
 __bss_end__ = .;
 . = ALIGN(8);
 . = . + 0x1000; /* 4kB of stack memory */
 stack_top = .;
 . = . + 0x1000;
 end = .;
}