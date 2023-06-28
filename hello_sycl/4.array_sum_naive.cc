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
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    int result = 0;

    {
        sycl::buffer<int32_t, 1> buf(arr.data(), sycl::range<1>(arr.size()));
        sycl::buffer<int32_t, 1> result_buf(&result, sycl::range<1>(1));

        sycl::queue queue(sycl::host_selector{});

        queue.submit([&](sycl::handler& cgh) {
            auto buff_acc = buf.get_access<sycl::access::mode::read>(cgh);
            // NOTE: accessor 还可以用来做数据的同步访问
            auto result_acc =
                result_buf.get_access<sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<class reduction_kernel>(
                sycl::range<1>(N),
                [=](sycl::item<1> id) { result_acc[0] += buff_acc[id]; });
        });
    }

    std::cout << "Sum: " << result << std::endl;

    return 0;
}
