#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_vector_add_shared;

class kernel_vector_add {
   public:
    kernel_vector_add(float *device_c, float *device_a, float *device_b)
        : device_c(device_c), device_a(device_a), device_b(device_b) {}

    void operator()() {
        device_c[0] = device_c[0] + this->device_a[0] + this->device_b[0];
        device_c[1] = device_c[1] + this->device_a[1] + this->device_b[1];
        device_c[2] = device_c[2] + this->device_a[2] + this->device_b[2];
        device_c[3] = device_c[3] + this->device_a[3] + this->device_b[3];
    }

   private:
    float *device_c, *device_a, *device_b;
};

class kernel_vector_copy_a {
   public:
    kernel_vector_copy_a(float *device_c, float *device_a, float *device_b)
        : device_c(device_c), device_a(device_a), device_b(device_b) {}

    void operator()() {
        device_c[0] = this->device_a[0];
        device_c[1] = this->device_a[1];
        device_c[2] = this->device_a[2];
        device_c[3] = this->device_a[3];
    }

   private:
    float *device_c, *device_a, *device_b;
};

void hello_usm_device() {
    // <<Setup host storage>>
    sycl::vec<float, 4> a = {1.0, 2.0, 3.0, 4.0};
    sycl::vec<float, 4> b = {1.0, 2.0, 3.0, 4.0};
    sycl::vec<float, 4> c = {0.0, 0.0, 0.0, 0.0};
    sycl::host_selector device;
    sycl::queue queue(device);
    std::cout << "Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    float *device_a = sycl::malloc_device<float>(4, queue);
    float *device_b = sycl::malloc_device<float>(4, queue);
    float *device_c = sycl::malloc_device<float>(4, queue);

    queue.memcpy(device_a, &a, 16).wait();
    queue.memcpy(device_b, &b, 16).wait();
    queue.memcpy(device_c, &c, 16).wait();

    // NOTE: 由于 kernel_vector_add 和 kernel_vector_copy_a 并没有使用 accessor
    // 来声明它们的依赖关系, 所以虽然实际上它们对 device_c 有 write-write data
    // race, 但 sycl runtime 并不知道...实际执行时结果有可能是 [1,2,3,4] 或
    // [2,4,6,8], 或者可能是 udefined behaviour?
    //
    // NOTE: 为了解决上面的问题, 可以使用 queue.wait, event.depends_on 以及
    // queue::in_order 属性, 例如
    // sycl::queue(device,sycl::property::queue::in_order{})
    //
    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task(kernel_vector_add(device_c, device_a, device_b));
    });

    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task(kernel_vector_copy_a(device_c, device_a, device_b));
    });

    queue.wait();
    queue.memcpy(&c, device_c, 16).wait();
    // <<Print results>>
    std::cout << c.x() << "," << c.y() << "," << c.z() << "," << c.w()
              << std::endl;
}

void hello_usm_shared() {
    // <<Setup host storage>>
    sycl::vec<float, 4> a = {1.0, 2.0, 3.0, 4.0};
    sycl::vec<float, 4> b = {1.0, 2.0, 3.0, 4.0};
    sycl::host_selector device;
    sycl::queue queue(device);

    float *device_a = sycl::malloc_device<float>(4, queue);
    float *device_b = sycl::malloc_device<float>(4, queue);
    float *c = sycl::malloc_shared<float>(4, queue);

    queue.memcpy(device_a, &a, 16).wait();
    queue.memcpy(device_b, &b, 16).wait();

    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class kernel_vector_add_shared>([=]() {
            c[0] = device_a[0] + device_b[0];
            c[1] = device_a[1] + device_b[1];
            c[2] = device_a[2] + device_b[2];
            c[3] = device_a[3] + device_b[3];
        });
    });

    queue.wait();
    std::cout << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << std::endl;
}

int main(int argc, char *argv[]) {
    hello_usm_device();
    hello_usm_shared();
    return 0;
}
