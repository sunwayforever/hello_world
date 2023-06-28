#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>
#include <time.h>

#include <cstdlib>
#include <iostream>

#include "julia_calc.h"

constexpr size_t SIZE = 2000;

class JuliaApp : public ci::app::App {
    // Texture for displaying the set
    ci::gl::Texture2dRef tex_;
    void* data_;
    JuliaCalculator* calc_;
    float center_x_, center_y_;
    float zoom_;

   public:
    JuliaApp()
        : data_(malloc(SIZE * SIZE * 4)),
          calc_(JuliaCalculator::get(SIZE, data_)),
          zoom_(1.0),
          center_x_(0.0),
          center_y_(0.0) {
        srand(time(NULL));
    }

    void setup() override {
        this->tex_ = ci::gl::Texture2d::create(
            nullptr, GL_RGBA, SIZE, SIZE,
            ci::gl::Texture2d::Format()
                .dataType(GL_UNSIGNED_BYTE)
                .internalFormat(GL_RGBA));
    }

    void update() override { calc_->Calc(); }

    void draw() override {
        ci::gl::clear();
        tex_->update(data_, GL_RGBA, GL_UNSIGNED_BYTE, 0, SIZE, SIZE);
        ci::Rectf screen(
            0, 0, getWindow()->getWidth(), getWindow()->getHeight());
        ci::gl::draw(tex_, screen);
    }

    void mouseWheel(ci::app::MouseEvent event) override {
        auto inc = event.getWheelIncrement();
        if (inc > 0) {
            zoom_ *= 1.1;
        } else {
            zoom_ /= 1.1;
        }
        calc_->SetZoom(zoom_);
    }

    void mouseDown(ci::app::MouseEvent event) override {
        if (event.isControlDown()) {
            calc_->SetC(getRandomFloat(), getRandomFloat());
            return;
        }
        if (event.isShiftDown()) {
            zoom_ = 1.0;
            calc_->SetZoom(zoom_);
            center_x_ = 0.0;
            center_y_ = 0.0;
            calc_->SetCenter(center_y_, center_x_);
            return;
        }
        auto w_width = getWindow()->getWidth();
        auto w_height = getWindow()->getHeight();

        auto x = (event.getX() - 0.5 * w_width) / (0.5 * w_width);
        auto y = (event.getY() - 0.5 * w_height) / (0.5 * w_height);

        center_x_ -= x / zoom_;
        center_y_ -= y / zoom_;

        calc_->SetCenter(center_y_, -center_x_);
    }

   private:
    float getRandomFloat() { return (rand() / (RAND_MAX + 1.0)) * 2.0 - 1.0; }
};

CINDER_APP(JuliaApp, ci::app::RendererGl(ci::app::RendererGl::Options{}))
