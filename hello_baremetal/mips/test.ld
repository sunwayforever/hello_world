ENTRY(_Reset)
SECTIONS
{
 . = 0x80100400;
 .startup . : { startup.o(.text) }
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
