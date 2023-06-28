// 2022-08-22 13:55
#ifndef MSPACE_CONFIG_H
#define MSPACE_CONFIG_H
#include <stddef.h>

static char buffer_1[1024000];
static char buffer_2[102400000];

void* SPACES[] = {
    buffer_1,
    buffer_2,
};

size_t SPACE_SIZES[] = {
    sizeof(buffer_1),
    sizeof(buffer_2),
};
int N_SPACES = sizeof(SPACES) / sizeof(SPACES[0]);

#endif  // MSPACE_CONFIG_H
