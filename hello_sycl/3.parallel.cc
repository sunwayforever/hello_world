#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_dummy;

void rot13(sycl::queue &queue, char *text) {
    int N = strlen(text);
    sycl::buffer<char, 1> buf(text, sycl::range<1>(N));
    queue.submit([&](sycl::handler &handle) {
        auto buf_acc = buf.get_access<sycl::access::mode::read_write>(handle);
        handle.parallel_for<class kernel_dummy>(
            sycl::range<1>(N), [=](sycl::item<1> item) {
                size_t id = item.get_linear_id();
                char c = buf_acc[id];
                buf_acc[id] = (c - 1 / (~(~c | 32) / 13 * 2 - 11) * 13);
            });
    });
}

int main(int argc, char *argv[]) {
    char text[] = "Hello World";
    sycl::queue queue(sycl::host_selector{}, [](sycl::exception_list el) {
        for (auto ex : el) {
            try {
                std::rethrow_exception(ex);
            } catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    });
    rot13(queue, text);
    printf("%s\n", text);
    rot13(queue, text);
    printf("%s\n", text);
    return 0;
}
