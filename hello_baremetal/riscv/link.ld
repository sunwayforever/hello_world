OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
  # NOTE: spike 默认的内存区域是 [0x80000000, 0x80000000*2]
  # 通过 spike --dump-dts main.elf 可以看到.
  # 通过 spike -m<a>:<b> 可以配置成其它区域
  . = 0x80000000;
  .text : { *(.text) }
  . = ALIGN(0x1000);
  .data : { *(.data) }
  .bss : { *(.bss) }
  _end = .;
}

