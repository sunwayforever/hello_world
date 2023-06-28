#include "julia_calc.h"

#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

sycl::float2 complex_mul(sycl::float2 a, sycl::float2 b) {
    return {a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x()};
}

sycl::float2 complex_add(sycl::float2 a, sycl::float2 b) {
    return {a.x() + b.x(), a.y() + b.y()};
}

float complex_norm(sycl::float2 a) { return a.x() * a.x() + a.y() * a.y(); }

int HowManySteps(sycl::float2 z, sycl::float2 c) {
    static constexpr size_t MAX_ITERS = 255;
    static constexpr float DIVERGENCE_LIMIT = 2.0;
    for (size_t i = MAX_ITERS; i > 0; i--) {
        z = complex_mul(z, z);
        z = complex_add(z, c);
        float norm = complex_norm(z);
        if (norm >= DIVERGENCE_LIMIT) {
            return i;
        }
    }
    return 0;
}

void JuliaCalculatorSycl::Calc() {
    queue_.submit([&](sycl::handler& cgh) {
        auto img_acc = img_.get_access<sycl::access::mode::read_write>(cgh);

        // NOTE: 这样写是因为 kernel 无法 capture this...
        int size = size_;
        float zoom = zoom_;
        float cx = cx_;
        float cy = cy_;
        float center_x = center_x_;
        float center_y = center_y_;
        cgh.parallel_for<class JuliaCalculator>(
            sycl::range<2>(size, size), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                float zx = (x - 0.5 * size) / (0.5 * size * zoom) + center_x;
                float zy =
                    (y - 0.5 * size) / (0.5 * size * zoom) + center_y;

                int count =
                    HowManySteps(sycl::float2{zx, zy}, sycl::float2{cx, cy});
                int color = (count << 21) + (count << 10) + (count << 3);
                img_acc[item] = {
                    (uint8_t)(color >> 16), (uint8_t)(color >> 8),
                    (uint8_t)color, (uint8_t)255};
            });
        cgh.copy(img_acc, data_);
    });
    // auto host_acc = img_.get_access<sycl::access::mode::read>();
    // memcpy(data_, host_acc.get_pointer(), width_ * height_ * 4);
    queue_.wait();
}

JuliaCalculator* JuliaCalculator::get(size_t size, void* data) {
    return new JuliaCalculatorSycl(size, data);
}
