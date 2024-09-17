#!/bin/bash
cat >/tmp/gdb.cmds <<hello
target remote localhost:1234
b _start_el1
hello

set -x
qemu-system-aarch64 -M virt -cpu cortex-a53 -kernel hello.elf -nographic -device loader,addr=0x40000000,cpu-num=0 -s -S &

qemu_pid=$!

gdb-multiarch hello.elf -x /tmp/gdb.cmds

kill -9 $qemu_pid &>/dev/null
