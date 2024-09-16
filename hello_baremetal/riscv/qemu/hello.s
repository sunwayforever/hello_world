.section .text
.global _start

## 初始运行在 M 模式
_start:
    ## 一般情况下 mepc 由 S 模式下的 ecall 设置, 这里直接设置了
    ## mepc.
    la      t0, supervisor
    csrw    mepc, t0
    ## 设置 mtvec 为 m_trap
    la      t1, m_trap
    csrw    mtvec, t1
    li      t2, 0x1800
    csrc    mstatus, t2
    li      t3, 0x800
    csrs    mstatus, t3
    li      t4, 0x100
    ## 0x100 设置为 mdeleg 表示 ecall 被 delegate 给 S 模式
    csrs    medeleg, t4
    ## mret 会降权为 S, 同时跳转到 mepc 即 supervisor
    mret

m_trap:
    ## mepc 会自动被设置为后面 s_trap::ecall+4 的地址, 以便 ecall 能返回
    csrr    t0, mepc
    ## mcause 为 9, 表示 Environment call from S-mode
    csrr    t1, mcause
    ## 这里手动设置 mepc 为 supervisor
    la      t2, supervisor
    csrw    mepc, t2
    ## mret 降权为 S 并跳转到 supervisor
    mret

supervisor:
    ## 设置 sepc 为 user
    la      t0, user
    csrw    sepc, t0
    ## 设置 stvec 为 s_trap
    la      t1, s_trap
    csrw    stvec, t1
    ## sret 会降权为 U 并跳转到 sepc 即 user
    sret

s_trap:
    csrr    t0, sepc
    ## scause 为 8, 表示 Environment call from U-mode
    csrr    t1, scause
    ## ecall 提权到 M 并调用 mtvec 即 m_trap
    ecall

user:
    csrr    t0, instret
    ## ecall 会提权到 S 并调用 stvec 即 s_trap
    ecall
