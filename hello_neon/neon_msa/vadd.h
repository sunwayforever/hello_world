// 2023-04-14 12:20
#ifndef VADD_H
#define VADD_H

#include <neon_emu_types.h>

#include "util.h"

// NOTE: msa 只支持 128 bit, 所以 neon 中非 `q` 类型的指令会浪费一倍的计算资源,
// 因为后 64 bit 并不需要处理
int8x8_t vadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_addv_b(a.v.i8, b.v.i8);
    return r;
}

int8x16_t vaddq_s8(int8x16_t a, int8x16_t b) {
    int8x16_t r;
    r.v.i8 = __msa_addv_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vaddhn_s16(int16x8_t a, int16x8_t b) {
    // NOTE: msa 不支持一般的 (a+b)>>c, 为了避免 a+b 时溢出, 只能使用更宽的数据
    // 类型,例如 int8 转换为 int16. 如果单个向量寄存器无法支持更宽的类型 (例如
    // int16x8 无法变成 int32x8), 则可能需要拆分成两个操作
    //
    // 另外, msa 不支持 int vector 的 mov 指令, 所以只能用
    // COPY 来模拟
    int16x8_t tmp;
    tmp.v.i16 = __msa_addv_h(a.v.i16, b.v.i16);
    tmp.v.i16 = __msa_srai_h(tmp.v.i16, 8);

    int8x8_t r;
    COPY(r, tmp);
    return r;
}

int8x8_t vhadd_s8(int8x8_t a, int8x8_t b) {
    // int16x8_t _a, _b, tmp;
    // COPY(_a, a);
    // COPY(_b, b);
    // tmp.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    // tmp.v.i16 = __msa_srai_h(tmp.v.i16, 1);
    int8x8_t r;
    r.v.i8 = __msa_ave_s_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vrhadd_s8(int8x8_t a, int8x8_t b) {
    // int16x8_t _a, _b, tmp;
    // COPY(_a, a);
    // COPY(_b, b);
    // tmp.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    // tmp.v.i16 = __msa_addvi_h(tmp.v.i16, 1);
    // tmp.v.i16 = __msa_srai_h(tmp.v.i16, 1);
    int8x8_t r;
    r.v.i8 = __msa_aver_s_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vqadd_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_adds_s_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vuqadd_s8(int8x8_t a, uint8x8_t b) {
    int8x8_t r;
    int16x8_t _a, _b, _r;
    COPY(_a, a);
    COPY(_b, b);
    _r.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    _r.v.i16 = __msa_sat_s_h(_r.v.i16, 7);
    COPY(r, _r);
    return r;
}

uint8x8_t vsqadd_u8(uint8x8_t a, int8x8_t b) {
    uint8x8_t r;
    int16x8_t _a, _b, _r;
    COPY(_a, a);
    COPY(_b, b);
    _r.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    // NOTE: msa 不存在类似的 saturating 操作
    for (int i = 0; i < 8; i++) {
        if (_r.values[i] > UINT8_MAX) {
            _r.values[i] = UINT8_MAX;
        }
        if (_r.values[i] < 0) {
            _r.values[i] = 0;
        }
    }
    COPY(r, _r);
    return r;
}

int8_t vqaddb_s8(int8_t a, int8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > INT8_MAX) {
        r = INT8_MAX;
    }
    if (r < INT8_MIN) {
        r = INT8_MIN;
    }
    return (int8_t)r;
}

int8_t vuqaddb_s8(int8_t a, uint8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > INT8_MAX) {
        r = INT8_MAX;
    }
    if (r < INT8_MIN) {
        r = INT8_MIN;
    }
    return (int8_t)r;
}

uint8_t vsqaddb_u8(uint8_t a, int8_t b) {
    int16_t r = (int16_t)a + b;
    if (r > UINT8_MAX) {
        r = UINT8_MAX;
    }
    if (r < 0) {
        r = 0;
    }
    return (uint8_t)r;
}

int16x8_t vaddl_s8(int8x8_t a, int8x8_t b) {
    int16x8_t r;
    int16x8_t _a, _b;
    COPY(_a, a);
    COPY(_b, b);
    r.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    return r;
}

int16x8_t vaddl_high_s8(int8x16_t a, int8x16_t b) {
    int16x8_t r;
    int16x8_t _a, _b;
    // NOTE: msa 无法支持 high 类型的指令, 它要求输入输出的的元素个数总是相同的
    COPY_HIGH(_a, a);
    COPY_HIGH(_b, b);
    r.v.i16 = __msa_addv_h(_a.v.i16, _b.v.i16);
    return r;
}

int16x8_t vaddw_s8(int16x8_t a, int8x8_t b) {
    int16x8_t r;
    int16x8_t _b;
    // NOTE: msa 基本不支持 widen/narrow 类型的指令, 它要求输入的类型是总是相同
    // 的 (除了 hadd/hsub)
    COPY(_b, b);
    r.v.i16 = __msa_addv_h(a.v.i16, _b.v.i16);
    return r;
}

int16x8_t vaddw_high_s8(int16x8_t a, int8x16_t b) {
    int16x8_t r;
    int16x8_t _b;
    COPY_HIGH(_b, b);
    r.v.i16 = __msa_addv_h(a.v.i16, _b.v.i16);
    return r;
}

#endif  // VADD_H
