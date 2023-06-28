// 2023-04-19 10:45
#ifndef VSHL_H
#define VSHL_H
// NOTE: neon 的 vector shift 都是通过 vshl 实现的, 其中 shift 为负时表示 right
// shift, 且 shift 需要先 saturate, 这个行为与 msa 的完全不一致: msa 不支持负数
// 的 shift, 且 shift 需要先 modulo
#endif  // VSHL_H
