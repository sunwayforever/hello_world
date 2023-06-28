#ifndef JULIA_CALC_CU_H
#define JULIA_CALC_CU_H

#include <iostream>

#include "../julia_calc.h"

class JuliaCalculatorCu : public JuliaCalculator {
   public:
    JuliaCalculatorCu(size_t size, void* data)
        : JuliaCalculator(size, data) {}

    void Calc();
};

#endif  // JULIA_CALC_SYCL_H
