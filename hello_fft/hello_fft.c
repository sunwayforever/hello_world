// 2022-09-27 15:24
#include <bits/pthreadtypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kissfft/kiss_fft.h"

#define PI 3.14159

#define TIMEIT_REPS 30
#define N 128

/* https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
 * https://zhuanlan.zhihu.com/p/538891353
 */

#define TIMEIT(F)                               \
    {                                           \
        clock_t start = clock();                \
        for (int i = 0; i < TIMEIT_REPS; ++i) { \
            F;                                  \
        }                                       \
        clock_t diff = clock() - start;         \
        printf("%s: %ld\n", #F, diff);          \
    }

void my_dft(int n_point, kiss_fft_cpx *in, kiss_fft_cpx *out) {
    for (int i = 0; i < n_point; i++) {
        float tmp_r = 0.0f;
        float tmp_i = 0.0f;
        for (int k = 0; k < n_point; k++) {
            tmp_r += in[k].r * cos(i * 2.0 * PI * k / n_point);
            tmp_i -= in[k].r * sin(i * 2.0 * PI * k / n_point);
        }
        out[i].r = tmp_r;
        out[i].i = tmp_i;
    }
}

void my_idft(int n_point, kiss_fft_cpx *in, kiss_fft_cpx *out) {
    for (int i = 0; i < n_point; i++) {
        float tmp = 0.0f;
        for (int k = 0; k < n_point; k++) {
            tmp += in[k].r * cos(2.0 * PI * i * k / n_point);
            tmp -= in[k].i * sin(2.0 * PI * i * k / n_point);
        }
        out[i].r = tmp;
        out[i].i = 0.0f;
    }
}

void my_recursive_fft(int n_point, kiss_fft_cpx *in, kiss_fft_cpx *out) {
    if (n_point == 1) {
        out[0].r = in[0].r;
        out[0].i = in[0].i;
        return;
    }
    kiss_fft_cpx *even_in =
        (kiss_fft_cpx *)KISS_FFT_MALLOC((n_point / 2) * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *even_out =
        (kiss_fft_cpx *)KISS_FFT_MALLOC((n_point / 2) * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *odd_in =
        (kiss_fft_cpx *)KISS_FFT_MALLOC((n_point / 2) * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *odd_out =
        (kiss_fft_cpx *)KISS_FFT_MALLOC((n_point / 2) * sizeof(kiss_fft_cpx));

    for (int i = 0; i < n_point / 2; i++) {
        even_in[i] = in[i * 2];
        odd_in[i] = in[i * 2 + 1];
    }
    my_recursive_fft(n_point / 2, even_in, even_out);
    my_recursive_fft(n_point / 2, odd_in, odd_out);

    int mid = n_point / 2;
    for (int i = 0; i < mid; i++) {
        float twiddle_factor_r = cos(-2 * PI * i / n_point);
        float twiddle_factor_i = sin(-2 * PI * i / n_point);
        out[i].r = even_out[i].r + odd_out[i].r * twiddle_factor_r -
                   odd_out[i].i * twiddle_factor_i;
        out[i].i = even_out[i].i + odd_out[i].r * twiddle_factor_i +
                   odd_out[i].i * twiddle_factor_r;

        twiddle_factor_r = cos(-2 * PI * (i + mid) / n_point);
        twiddle_factor_i = sin(-2 * PI * (i + mid) / n_point);
        out[i + mid].r = even_out[i].r + odd_out[i].r * twiddle_factor_r -
                         odd_out[i].i * twiddle_factor_i;
        out[i + mid].i = even_out[i].i + odd_out[i].r * twiddle_factor_i +
                         odd_out[i].i * twiddle_factor_r;
    }
}

extern float twiddle_table[];
void my_fft(
    int n_point, kiss_fft_cpx *in, kiss_fft_cpx *out, int with_twiddle_table) {
    int rev[n_point];
    for (int i = 0; i < n_point; i++) {
        rev[i] = 0;
    }
    int bit = (int)log2(n_point);
    for (int i = 1; i < n_point; i++) {
        rev[i] = (rev[i >> 1] >> 1 | ((i & 1) << (bit - 1)));
    }
    memcpy(out, in, n_point * sizeof(in[0]));
    for (int i = 1; i < n_point; i++) {
        if (i < rev[i]) {
            kiss_fft_cpx tmp = out[i];
            out[i] = out[rev[i]];
            out[rev[i]] = tmp;
        }
    }
    for (int mid = 1; mid < n_point; mid *= 2) {
        for (int j = 0; j < n_point; j += mid * 2) {
            for (int i = j; i < j + mid; i++) {
                kiss_fft_cpx even = out[i];
                kiss_fft_cpx odd = out[i + mid];

                int index = (int)log2(mid);
                float twiddle_factor_r = 0.0f;
                float twiddle_factor_i = 0.0f;
                if (with_twiddle_table) {
                    twiddle_factor_r = twiddle_table[index * N * 2 + i * 2];
                    twiddle_factor_i = twiddle_table[index * N * 2 + i * 2 + 1];
                } else {
                    twiddle_factor_r = cos(-1 * PI * i / mid);
                    twiddle_factor_i = sin(-1 * PI * i / mid);
                }
                out[i].r = even.r + odd.r * twiddle_factor_r -
                           odd.i * twiddle_factor_i;
                out[i].i = even.i + odd.r * twiddle_factor_i +
                           odd.i * twiddle_factor_r;

                if (with_twiddle_table) {
                    twiddle_factor_r =
                        twiddle_table[index * N * 2 + (i + mid) * 2];
                    twiddle_factor_i =
                        twiddle_table[index * N * 2 + (i + mid) * 2 + 1];
                } else {
                    twiddle_factor_r = cos(-1 * PI * (i + mid) / mid);
                    twiddle_factor_i = sin(-1 * PI * (i + mid) / mid);
                }
                out[i + mid].r = even.r + (odd.r * twiddle_factor_r -
                                           odd.i * twiddle_factor_i);
                out[i + mid].i = even.i + (odd.r * twiddle_factor_i +
                                           odd.i * twiddle_factor_r);
            }
        }
    }
}

extern int32_t twiddle_table_fixed[];
typedef struct {
    int32_t r;
    int32_t i;
} kiss_fft_cpx_fixed;

static inline int32_t MUL_Q15(int32_t a, int32_t b) {
    return ((int64_t)a * (int64_t)b) >> 15;
}

void my_fft_fixed(int n_point, kiss_fft_cpx *in, kiss_fft_cpx *out) {
    int rev[n_point];
    for (int i = 0; i < n_point; i++) {
        rev[i] = 0;
    }
    int bit = (int)log2(n_point);
    for (int i = 1; i < n_point; i++) {
        rev[i] = (rev[i >> 1] >> 1 | ((i & 1) << (bit - 1)));
    }
    memcpy(out, in, n_point * sizeof(in[0]));
    for (int i = 1; i < n_point; i++) {
        if (i < rev[i]) {
            kiss_fft_cpx tmp = out[i];
            out[i] = out[rev[i]];
            out[rev[i]] = tmp;
        }
    }
    /* convert out to fixed point */
    kiss_fft_cpx_fixed *out_fixed =
        malloc(n_point * sizeof(kiss_fft_cpx_fixed));
    for (int i = 0; i < n_point; i++) {
        out_fixed[i].r = (int32_t)(out[i].r * (1 << 15));
        out_fixed[i].i = (int32_t)(out[i].i * (1 << 15));
    }

    for (int mid = 1; mid < n_point; mid *= 2) {
        for (int j = 0; j < n_point; j += mid * 2) {
            for (int i = j; i < j + mid; i++) {
                kiss_fft_cpx_fixed even = out_fixed[i];
                kiss_fft_cpx_fixed odd = out_fixed[i + mid];

                int index = (int)log2(mid);
                int32_t twiddle_factor_r = 0;
                int32_t twiddle_factor_i = 0;
                twiddle_factor_r = twiddle_table_fixed[index * N * 2 + i * 2];
                twiddle_factor_i =
                    twiddle_table_fixed[index * N * 2 + i * 2 + 1];
                out_fixed[i].r = even.r + MUL_Q15(odd.r, twiddle_factor_r) -
                                 MUL_Q15(odd.i, twiddle_factor_i);
                out_fixed[i].i = even.i + MUL_Q15(odd.r, twiddle_factor_i) +
                                 MUL_Q15(odd.i, twiddle_factor_r);

                twiddle_factor_r =
                    twiddle_table_fixed[index * N * 2 + (i + mid) * 2];
                twiddle_factor_i =
                    twiddle_table_fixed[index * N * 2 + (i + mid) * 2 + 1];
                out_fixed[i + mid].r = even.r +
                                       MUL_Q15(odd.r, twiddle_factor_r) -
                                       MUL_Q15(odd.i, twiddle_factor_i);

                out_fixed[i + mid].i = even.i +
                                       MUL_Q15(odd.r, twiddle_factor_i) +
                                       MUL_Q15(odd.i, twiddle_factor_r);
            }
        }
    }
    /* convert back to float */
    for (int i = 0; i < n_point; i++) {
        out[i].r = out_fixed[i].r >> 15;
        out[i].i = out_fixed[i].i >> 15;
    }
}

void generate_twiddle_table(int n_point) {
    int bit = (int)log2(n_point);
    float twiddle_table[bit][n_point * 2];
    int32_t twiddle_table_fixed[bit][n_point * 2];
    for (int mid = 1; mid < n_point; mid *= 2) {
        for (int j = 0; j < n_point; j += mid * 2) {
            for (int i = j; i < j + mid; i++) {
                int index = (int)log2(mid);
                twiddle_table[index][i * 2] = cos(-1 * PI * i / mid);
                twiddle_table[index][i * 2 + 1] = sin(-1 * PI * i / mid);
                twiddle_table[index][(i + mid) * 2] =
                    cos(-1 * PI * (i + mid) / mid);
                twiddle_table[index][(i + mid) * 2 + 1] =
                    sin(-1 * PI * (i + mid) / mid);

                twiddle_table_fixed[index][i * 2] =
                    (int)(twiddle_table[index][i * 2] * (1 << 15));
                twiddle_table_fixed[index][i * 2 + 1] =
                    (int)(twiddle_table[index][i * 2 + 1] * (1 << 15));
                twiddle_table_fixed[index][(i + mid) * 2] =
                    (int)(twiddle_table[index][(i + mid) * 2] * (1 << 15));
                twiddle_table_fixed[index][(i + mid) * 2 + 1] =
                    (int)(twiddle_table[index][(i + mid) * 2 + 1] * (1 << 15));
            }
        }
    }
    FILE *fp = fopen("twiddle_table.c", "w");
    fprintf(fp, "float twiddle_table[%d*%d] = {\n", bit, n_point * 2);
    for (int i = 0; i < bit * n_point * 2; i++) {
        fprintf(fp, "%f,", ((float *)twiddle_table)[i]);
    }
    fprintf(fp, "};\n");

    fprintf(fp, "#include <stdint.h>\n");
    fprintf(fp, "int32_t twiddle_table_fixed[%d*%d] = {\n", bit, n_point * 2);
    for (int i = 0; i < bit * n_point * 2; i++) {
        fprintf(fp, "%d,", ((int32_t *)twiddle_table_fixed)[i]);
    }
    fprintf(fp, "};\n");
    fclose(fp);
}

void clear(kiss_fft_cpx *data) {
    for (int i = 0; i < N; i++) {
        data[i].r = 0.0f;
        data[i].i = 0.0f;
    }
}

int main(int argc, char *argv[]) {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, 0, 0);
    kiss_fft_cpx *in =
        (kiss_fft_cpx *)KISS_FFT_MALLOC(N * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *out =
        (kiss_fft_cpx *)KISS_FFT_MALLOC(N * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *in_restored =
        (kiss_fft_cpx *)KISS_FFT_MALLOC(N * sizeof(kiss_fft_cpx));
    /* NOTE: 1hz cos with amplitude 1 + 2hz sin with amplitude 10 */
    for (int i = 0; i < N; i++) {
        in[i].r = cos(1 * i * 2.0 * PI / N) + 10 * sin(2 * i * 2.0 * PI / N);
    }
    clear(out);
    TIMEIT(kiss_fft(cfg, in, out));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(out);
    TIMEIT(my_dft(N, in, out));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(out);
    TIMEIT(my_recursive_fft(N, in, out));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(out);
    TIMEIT(my_fft(N, in, out, 0));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(out);
    TIMEIT(my_fft(N, in, out, 1));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(out);
    TIMEIT(my_fft_fixed(N, in, out));
    for (int i = 0; i < 5; i++) {
        printf("[%.3f,%.3f]\n", out[i].r, out[i].i);
    }
    printf("------\n");

    clear(in_restored);
    cfg = kiss_fft_alloc(N, 1, 0, 0);
    TIMEIT(kiss_fft(cfg, out, in_restored));
    for (int i = 0; i < 5; i++) {
        printf("%f %f\n", in_restored[i].r / N, in[i].r);
    }
    printf("------\n");

    clear(in_restored);
    TIMEIT(my_idft(N, out, in_restored))
    for (int i = 0; i < 5; i++) {
        printf("%f %f\n", in_restored[i].r / N, in[i].r);
    }
    /* NOTE: uncoment this line to generate twiddle table for `my_fft` and
     * `my_fft_fixed` */
    /* generate_twiddle_table(N); */
}
