// 2023-04-14 19:39
#ifndef VMUL_H
#define VMUL_H

#include <neon_emu_types.h>

int8x8_t vmul_s8(int8x8_t a, int8x8_t b) {
    int8x8_t r;
    r.v.i8 = __msa_mulv_b(a.v.i8, b.v.i8);
    return r;
}

int8x16_t vmulq_s8(int8x16_t a, int8x16_t b) {
    int8x16_t r;
    r.v.i8 = __msa_mulv_b(a.v.i8, b.v.i8);
    return r;
}

int8x8_t vmla_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    r.v.i8 = __msa_addv_b(a.v.i8, __msa_mulv_b(b.v.i8, c.v.i8));
    return r;
}

int8x8_t vmls_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
    int8x8_t r;
    r.v.i8 = __msa_subv_b(a.v.i8, __msa_mulv_b(b.v.i8, c.v.i8));
    return r;
}

int16x8_t vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
    int16x8_t r;
    int16x8_t _b, _c;
    COPY(_b, b);
    COPY(_c, c);
    r.v.i16 = __msa_addv_h(a.v.i16, __msa_mulv_h(_b.v.i16, _c.v.i16));
    return r;
}

int16x8_t vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
    int16x8_t r;
    int16x8_t _b, _c;
    COPY(_b, b);
    COPY(_c, c);
    r.v.i16 = __msa_subv_h(a.v.i16, __msa_mulv_h(_b.v.i16, _c.v.i16));
    return r;
}

int16x8_t vmlal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
    int8x8_t _b, _c;
    COPY_HIGH(_b, b);
    COPY_HIGH(_c, c);
    return vmlal_s8(a, _b, _c);
}

int16x8_t vmlsl_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
    int8x8_t _b, _c;
    COPY_HIGH(_b, b);
    COPY_HIGH(_c, c);
    return vmlsl_s8(a, _b, _c);
}

float32x2_t vfma_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
    float32x2_t r;
    r.v.f32 = __msa_fmadd_w(a.v.f32, b.v.f32, c.v.f32);
    return r;
}

float32x2_t vfma_lane_f32(
    float32x2_t a, float32x2_t b, float32x2_t v, int lane) {
    float32x2_t r;
    float32x2_t _v;
    _v.values[0] = v.values[lane];
    _v.values[1] = v.values[lane];
    return vfma_f32(a, b, _v);
}

float32x2_t vfma_laneq_f32(
    float32x2_t a, float32x2_t b, float32x4_t v, int lane) {
    float32x2_t r;
    float32x2_t _v;
    _v.values[0] = v.values[lane];
    _v.values[1] = v.values[lane];
    return vfma_f32(a, b, _v);
}

float vfmas_lane_f32(float a, float b, float32x2_t v, int lane) {
    float r = (double)a + (double)b * v.values[lane];
    return r;
}

// NOTE: 如果 int16x4_t 换成 int32x4_t, 则无法支持
int16x4_t vqdmulh_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    int32x4_t _a, _b, _r, _two;
    COPY(_a, a);
    COPY(_b, b);
    _two.v.i32 = __msa_fill_w(2);
    _r.v.i32 = __msa_mulv_w(_a.v.i32, _b.v.i32);
    _r.v.i32 = __msa_mulv_w(_r.v.i32, _two.v.i32);
    _r.v.i32 = __msa_srai_w(_r.v.i32, 16);
    COPY(r, _r);
    return r;
}

int16x4_t vqrdmulh_s16(int16x4_t a, int16x4_t b) {
    int16x4_t r;
    int32x4_t _a, _b, _r, _two;
    COPY(_a, a);
    COPY(_b, b);
    _two.v.i32 = __msa_fill_w(2);
    _r.v.i32 = __msa_mulv_w(_a.v.i32, _b.v.i32);
    _r.v.i32 = __msa_mulv_w(_r.v.i32, _two.v.i32);
    _r.v.i32 = __msa_srari_w(_r.v.i32, 16);
    COPY(r, _r);
    return r;
}

int32x4_t vqdmull_s16(int16x4_t a, int16x4_t b) {
    int32x4_t r;
    int32x4_t _a, _b, _two;
    COPY(_a, a);
    COPY(_b, b);
    _two.v.i32 = __msa_fill_w(2);
    r.v.i32 = __msa_mulv_w(_a.v.i32, _b.v.i32);
    r.v.i32 = __msa_mulv_w(r.v.i32, _two.v.i32);
    return r;
}

int16x8_t vmull_s8(int8x8_t a, int8x8_t b) {
    int16x8_t r;
    int16x8_t _a, _b;
    COPY(_a, a);
    COPY(_b, b);
    r.v.i16 = __msa_mulv_h(_a.v.i16, _b.v.i16);
    return r;
}

int16x8_t vmull_high_s8(int8x16_t a, int8x16_t b) {
    int16x8_t r;
    int8x8_t _a, _b;
    COPY_HIGH(_a, a);
    COPY_HIGH(_b, b);
    return vmull_s8(_a, _b);
}

int32x4_t vmull_n_s16(int16x4_t a, int16_t b) {
    int32x4_t r;
    int32x4_t _a, _b;
    _b.v.i32 = __msa_fill_w(b);
    r.v.i32 = __msa_mulv_w(_a.v.i32, _b.v.i32);
    return r;
};

poly8x8_t vmul_p8(poly8x8_t a, poly8x8_t b) {
    poly8x8_t r;
    for (int i = 0; i < 8; i++) {
        uint16_t tmp = 0;
        uint8_t x = a.values[i];
        uint8_t y = b.values[i];
        for (int j = 0; j < 8; j++) {
            // x=1111
            // y=1010
            if (y & 1) {
                tmp ^= x;
            }
            y >>= 1;
            x <<= 1;
        }
        r.values[i] = (uint8_t)tmp;
    }
    return r;
};
#endif  // VMUL_H
