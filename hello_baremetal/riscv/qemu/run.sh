#!/bin/bash
cat >/tmp/gdb.cmds <<hello
target remote localhost:1234
b _start
c
hello

qemu-system-riscv64 -M virt -bios hello.elf -nographic -s -S &

qemu_pid=$!

gdb-multiarch hello.elf -x /tmp/gdb.cmds

kill -9 $qemu_pid &>/dev/null
