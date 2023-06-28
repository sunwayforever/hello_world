// 2023-04-21 10:58
#include <neon.h>
#include <neon_test.h>

// int8x8_t vcreate_s8(uint64_t a)
// int16x4_t vcreate_s16(uint64_t a)
// int32x2_t vcreate_s32(uint64_t a)
// int64x1_t vcreate_s64(uint64_t a)
// uint8x8_t vcreate_u8(uint64_t a)
// uint16x4_t vcreate_u16(uint64_t a)
// uint32x2_t vcreate_u32(uint64_t a)
// uint64x1_t vcreate_u64(uint64_t a)
// poly64x1_t vcreate_p64(uint64_t a)
// float16x4_t vcreate_f16(uint64_t a)
// float32x2_t vcreate_f32(uint64_t a)
// poly8x8_t vcreate_p8(uint64_t a)
// poly16x4_t vcreate_p16(uint64_t a)
// float64x1_t vcreate_f64(uint64_t a)

TEST_CASE(test_vcreate_s8) {
    static const struct {
        uint64_t a[1];
    } test_vec[] = {
        {{14132917921477899950ull}}, {{9768881841052706856ull}},
        {{16325103149125810475ull}}, {{2241800239056659389ull}},
        {{16892050861247466928ull}}, {{6292462352927236486ull}},
        {{13564512221404632202ull}}, {{13980988618246101366ull}},
    };

    for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        int8x8_t a = vcreate_s8(test_vec[i].a[0]);
        uint64x1_t r = vreinterpret_u64_s8(a);
        uint64x1_t check = vld1_u64(test_vec[i].a);
        ASSERT_EQUAL(r, check);
    }
    return 0;
}
