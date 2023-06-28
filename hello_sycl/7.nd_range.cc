#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_dummy;
class kernel_dummy_2;
class kernel_dummy_3;
int main(int argc, char *argv[]) {
    // printf("range\n");
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
    queue.submit([&](sycl::handler &handle) {
        handle.parallel_for<class kernel_dummy>(
            // NOTE: 使用 range 时 sycl 会自己决定 workgroup/work_item 的个数,
            // 给用户的 item.get_id 相当于 nd_item 的 get_global_linear_id
            sycl::range<2>(2, 5), [=](sycl::item<2> item) {
                printf("%ld %ld\n", item.get_id(0), item.get_id(1));
            });
    });

    queue.wait_and_throw();

    printf("nd_range\n");
    // NOTE: 与 cuda 不同的是, work_group 大小是 work_group*work_item, 而不是
    // work_group 自己的 count, 所以: `两个 group, 每个 group 有 5 个 item` 是
    // nd_range<1>(10,5), 而不是 nd_range<1>(2,5)
    queue.submit([&](sycl::handler &handle) {
        handle.parallel_for<class kernel_dummy_2>(
            sycl::nd_range<1>(10, 5), [=](sycl::nd_item<1> item) {
                printf(
                    "global:%ld group:%ld local:%ld\n",
                    item.get_global_linear_id(), item.get_group_linear_id(),
                    item.get_local_linear_id());
            });
    });
    queue.wait_and_throw();

    printf("nd_range_2_dim\n");
    // NOTE: nd_range<2>(range<2>(4, 10), range<2>(2,5)) 表示的
    // group 与 item 的排列为:
    //          10
    //   +--------------
    //   |     5     5
    //   | 2 iiiii iiiii
    // 4 |   iiiii iiiii
    //   |
    //   | 2 iiiii iiiii
    //   |   iiiii iiiii
    //
    // 所以一个 2x2 个 group, 每个 group 有 2x5 的 item
    //
    // 与 cuda 不同的是 work_group 与 work_item 的 dim 需要是相同的, 所以如果需要指
    // 定类似于 cuda <<<2x3, 3>>> 的结构, 只能是 nd_range<2>(range<2>(2,9),range(1,3))
    queue.submit([&](sycl::handler &handle) {
        handle.parallel_for<class kernel_dummy_3>(
            sycl::nd_range<2>(sycl::range<2>(4, 5), sycl::range<2>(2, 5)),
            [=](sycl::nd_item<2> item) {
                printf(
                    "global:%ld global[0]:%ld global[1]:%ld | group:%ld "
                    "group[0]:%ld group[1]:%ld | local:%ld "
                    "local[0]:%ld "
                    "local[1]:%ld\n",
                    item.get_global_linear_id(), item.get_global_id(0),
                    item.get_global_id(1), item.get_group_linear_id(),
                    item.get_group(0), item.get_group(1),
                    item.get_local_linear_id(), item.get_local_id(0),
                    item.get_local_id(1));
            });
    });
    queue.wait_and_throw();
    return 0;
}
