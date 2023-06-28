#include "julia_calc.h"

extern void Julia(
    int size, float zoom, void* data, float cx, float cy, float center_x,
    float center_y);

void JuliaCalculatorCu::Calc() {
    Julia(size_, zoom_, data_, cx_, cy_, center_x_, center_y_);
}

JuliaCalculator* JuliaCalculator::get(size_t size, void* data) {
    return new JuliaCalculatorCu(size, data);
}
