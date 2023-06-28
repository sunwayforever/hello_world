// 2023-04-20 19:38
#ifndef VCOPY_LANE_H
#define VCOPY_LANE_H

#include <neon_emu_types.h>

int8x8_t vcopy_lane_s8(int8x8_t a, int lane_a, int8x8_t b, int lane_b) {
    int8x8_t r = a;
    r.values[lane_a] = b.values[lane_b];
    return r;
}
#endif  // VCOPY_LANE_H
