#ifndef POOL_CONFIG_64_H
#define POOL_CONFIG_64_H
#include <stddef.h>
size_t BIN_SIZE = 64;
static char buffer_1[61392];
static char buffer_2[6800];
static char buffer_3[4624];
static char buffer_4[3344];
static char buffer_5[2896];
static char buffer_6[3472];
static char buffer_7[2704];
static char buffer_8[2576];
static char buffer_9[2320];
static char buffer_10[3216];
static char buffer_11[3536];
static char buffer_12[2320];
static char buffer_13[3344];
static char buffer_14[3600];
static char buffer_15[2896];
static char buffer_16[2064];
static char buffer_18[3472];
static char buffer_20[1296];
static char buffer_21[1360];
static char buffer_22[1424];
static char buffer_23[1488];
static char buffer_24[4624];
static char buffer_26[3344];
static char buffer_28[3600];
static char buffer_30[1936];
static char buffer_32[2064];
static char buffer_33[2128];
static char buffer_37[2384];
static char buffer_39[2512];
static char buffer_40[2576];
static char buffer_49[3152];
static char buffer_52[3344];
static char buffer_53[3408];
static char buffer_54[3472];
static char buffer_58[3728];
static char buffer_72[4624];
static char buffer_89[5712];
static char buffer_93[5968];
static char buffer_101[6480];
static char buffer_128[8208];
static char buffer_133[8528];
static char buffer_136[8720];
static char buffer_487[31184];
void *BUFFERS[]={
buffer_1,buffer_2,buffer_3,buffer_4,buffer_5,buffer_6,buffer_7,buffer_8,buffer_9,buffer_10,buffer_11,buffer_12,buffer_13,buffer_14,buffer_15,buffer_16,buffer_18,buffer_20,buffer_21,buffer_22,buffer_23,buffer_24,buffer_26,buffer_28,buffer_30,buffer_32,buffer_33,buffer_37,buffer_39,buffer_40,buffer_49,buffer_52,buffer_53,buffer_54,buffer_58,buffer_72,buffer_89,buffer_93,buffer_101,buffer_128,buffer_133,buffer_136,buffer_487};
void *POOLS[]={
0,buffer_1,buffer_2,buffer_3,buffer_4,buffer_5,buffer_6,buffer_7,buffer_8,buffer_9,buffer_10,buffer_11,buffer_12,buffer_13,buffer_14,buffer_15,buffer_16,0,buffer_18,0,buffer_20,buffer_21,buffer_22,buffer_23,buffer_24,0,buffer_26,0,buffer_28,0,buffer_30,0,buffer_32,buffer_33,0,0,0,buffer_37,0,buffer_39,buffer_40,0,0,0,0,0,0,0,0,buffer_49,0,0,buffer_52,buffer_53,buffer_54,0,0,0,buffer_58,0,0,0,0,0,0,0,0,0,0,0,0,0,buffer_72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,buffer_89,0,0,0,buffer_93,0,0,0,0,0,0,0,buffer_101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,buffer_128,0,0,0,0,buffer_133,0,0,buffer_136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,buffer_487};
size_t BUFFER_CAPACITIES[]={
sizeof(buffer_1),sizeof(buffer_2),sizeof(buffer_3),sizeof(buffer_4),sizeof(buffer_5),sizeof(buffer_6),sizeof(buffer_7),sizeof(buffer_8),sizeof(buffer_9),sizeof(buffer_10),sizeof(buffer_11),sizeof(buffer_12),sizeof(buffer_13),sizeof(buffer_14),sizeof(buffer_15),sizeof(buffer_16),sizeof(buffer_18),sizeof(buffer_20),sizeof(buffer_21),sizeof(buffer_22),sizeof(buffer_23),sizeof(buffer_24),sizeof(buffer_26),sizeof(buffer_28),sizeof(buffer_30),sizeof(buffer_32),sizeof(buffer_33),sizeof(buffer_37),sizeof(buffer_39),sizeof(buffer_40),sizeof(buffer_49),sizeof(buffer_52),sizeof(buffer_53),sizeof(buffer_54),sizeof(buffer_58),sizeof(buffer_72),sizeof(buffer_89),sizeof(buffer_93),sizeof(buffer_101),sizeof(buffer_128),sizeof(buffer_133),sizeof(buffer_136),sizeof(buffer_487)};
size_t BUFFER_SIZES[]={
64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024,1152,1280,1344,1408,1472,1536,1664,1792,1920,2048,2112,2368,2496,2560,3136,3328,3392,3456,3712,4608,5696,5952,6464,8192,8512,8704,31168};
size_t BUFFER_COUNTS[]={
959,53,24,13,9,9,6,5,4,5,5,3,4,4,3,2,3,1,1,1,1,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
int N_BUFFER=43;
#endif //POOL_CONFIG_64_H