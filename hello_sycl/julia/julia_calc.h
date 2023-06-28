#ifndef JULIA_CALC_H
#define JULIA_CALC_H

#include <iostream>

class JuliaCalculator {
   protected:
    size_t const size_;
    float zoom_ = 1.0;
    void* data_;
    float cx_ = 0.285;
    float cy_ = 0.01;
    float center_x_ = 0.0;
    float center_y_ = 0.0;

   public:
    JuliaCalculator(size_t size, void* data) : size_(size), data_(data) {}

    virtual void Calc() = 0;

    void SetZoom(float zoom) { zoom_ = zoom; }

    void SetC(float x, float y) {
        cx_ = x;
        cy_ = y;
    }

    void SetCenter(float x, float y) {
        center_x_ = x;
        center_y_ = y;
    }

    static JuliaCalculator* get(size_t size, void* data);
};

#endif  // JULIA_CALC_H
