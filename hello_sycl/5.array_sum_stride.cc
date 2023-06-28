#include <CL/sycl.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

class reduction_kernel;
namespace sycl = cl::sycl;

#define N 100

int main(int, char**) {
    std::array<int32_t, N> arr;
    std::cout << "Data: ";
    for (int stride = 0; stride < N; stride++) {
        arr[stride] = stride + 1;
        std::cout << arr[stride] << " ";
    }
    std::cout << std::endl;

    {
        sycl::buffer<int32_t, 1> buf(arr.data(), sycl::range<1>(arr.size()));

        sycl::queue queue(sycl::host_selector{});
        for (int stride = 1; stride < N; stride *= 2) {
            queue.submit([&](sycl::handler& cgh) {
                auto buff_acc =
                    buf.get_access<sycl::access::mode::read_write>(cgh);
                cgh.parallel_for<class reduction_kernel>(
                    sycl::range<1>(N), [=](sycl::item<1> item) {
                        auto id = 2 * stride * item.get_linear_id();
                        if (id < N) {
                            buff_acc[id] = buff_acc[id] + buff_acc[id + stride];
                        }
                    });
            });
        }
    }

    std::cout << "Sum: " << arr[0] << std::endl;

    return 0;
}
