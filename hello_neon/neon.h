// 2023-04-14 10:48
#ifndef COMMON_H
#define COMMON_H

#ifdef __aarch64__
#include <arm_neon.h>
#else
#include <neon_emu.h>
#endif

#endif  // COMMON_H
