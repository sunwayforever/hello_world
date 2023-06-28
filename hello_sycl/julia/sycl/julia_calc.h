#ifndef JULIA_CALC_SYCL_H
#define JULIA_CALC_SYCL_H

#include <CL/sycl.hpp>
#include <iostream>

#include "../julia_calc.h"

namespace sycl = cl::sycl;

class JuliaCalculatorSycl : public JuliaCalculator {
    sycl::queue queue_;
    sycl::buffer<sycl::cl_uchar4, 2> img_;

   public:
    JuliaCalculatorSycl(size_t size, void* data)
        : JuliaCalculator(size, data),
          queue_(sycl::host_selector{}),
          img_(sycl::range<2>(size, size)) {}

    void Calc();
};

#endif  // JULIA_CALC_SYCL_H
