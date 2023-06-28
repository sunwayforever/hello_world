// 2023-04-19 16:24
#ifndef VSHR_H
#define VSHR_H

#include <neon_emu_types.h>
#include <stdint.h>

// NOTE: the `inline` is mandatory because `n` need to be a constant for
// __msa_srai_h, `inline` will enable CCP optimization (without lto)
//
// NOTE: __msa_srai_h immediate shift only support shift range of [0,15], while
// neon support [0, 16], so we may need to change __msa_srai_h to __msa_sra_h
inline int16x4_t vshr_n_s16(int16x4_t a, int n) {
    int16x4_t r;
    r.v.i16 = __msa_srai_h(a.v.i16, n);
    return r;
}

inline uint16x4_t vshr_n_u16(uint16x4_t a, int n) {
    uint16x4_t r;
    r.v.i16 = __msa_srli_h(a.v.i16, n);
    return r;
}

inline int8x8_t vrshr_n_s8(int8x8_t a, int n) {
    int8x8_t r;
    r.v.i8 = __msa_srari_b(a.v.i8, n);
    return r;
}

inline int8x8_t vsra_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    r.v.i8 = __msa_addv_b(a.v.i8, __msa_srai_b(b.v.i8, n));
    return r;
}

inline int8x8_t vrsra_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r;
    r.v.i8 = __msa_addv_b(a.v.i8, __msa_srari_b(b.v.i8, n));
    return r;
}

inline int8x8_t vshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    int16x8_t _r;
    _r.v.i16 = __msa_srai_h(a.v.i16, n);
    COPY(r, _r);
    return r;
}

inline int8x16_t vshrn_high_n_s16(int8x8_t a, int16x8_t b, int n) {
    int8x16_t r;
    int16x8_t tmp;
    tmp.v.i16 = __msa_srai_h(b.v.i16, n);
    MERGE(r, a, tmp);
    return r;
}

// NOTE: msa 的 sat 指令不支持把一个 signed int 按 unsigned int 的范围来
// saturate
uint8x8_t vqshrun_n_s16(int16x8_t a, int n) {
    uint8x8_t r;
    int16_t tmp;
    for (int i = 0; i < 8; i++) {
        tmp = a.values[i] >> n;
        if (tmp > UINT8_MAX) {
            tmp = UINT8_MAX;
        }
        if (tmp < 0) {
            tmp = 0;
        }
        r.values[i] = tmp;
    }
    return r;
}

inline int8x8_t vqshrn_n_s16(int16x8_t a, int n) {
    int16x8_t _r;
    _r.v.i16 = __msa_srai_h(a.v.i16, n);
    _r.v.i16 = __msa_sat_s_h(_r.v.i16, 7);
    int8x8_t r;
    COPY(r, _r);
    return r;
}

inline int8x8_t vqrshrn_n_s16(int16x8_t a, int n) {
    int16x8_t _r;
    _r.v.i16 = __msa_srari_h(a.v.i16, n);
    _r.v.i16 = __msa_sat_s_h(_r.v.i16, 7);
    int8x8_t r;
    COPY(r, _r);
    return r;
}

inline int8x8_t vrshrn_n_s16(int16x8_t a, int n) {
    int8x8_t r;
    int16x8_t _r;
    _r.v.i16 = __msa_srari_h(a.v.i16, n);
    COPY(r, _r);
    return r;
}

#include <vbsl.h>
inline int8x8_t vsri_n_s8(int8x8_t a, int8x8_t b, int n) {
    int8x8_t r, _b;
    uint8x8_t mask;
    _b.v.i8 = __msa_srai_b(b.v.i8, n);
    mask.v.i8 = __msa_fill_b((1 << (8 - n)) - 1);
    return vbsl_s8(mask, _b, a);
};
#endif  // VSHR_H
