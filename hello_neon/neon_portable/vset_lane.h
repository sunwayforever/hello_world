// 2023-04-21 16:34
#ifndef VSET_LANE_H
#define VSET_LANE_H

#include <neon_emu_types.h>

int8x8_t vset_lane_s8(int8_t a, int8x8_t v, int lane) {
    int8x8_t r = v;
    r.values[lane] = a;
    return r;
}
#endif  // VSET_LANE_H
