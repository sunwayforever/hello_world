#include <CL/sycl.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

class reduction_kernel;
namespace sycl = cl::sycl;

// 假设 array 大小为 128, 会分成两个 group, 每个 group 有 64 个数:
// 0 1 2 3 ... 63     64 65 66 ... 127
//
// 一共会启动 128 个线程, 初始时每个 group 的 32 个线程会进行一次 reduce:
//
// 0 1 2 3 ... 63     64 65 66 ... 127
// 1   5  ...         129   133 ...
//
// 初始 reduce 的结果写在 local memory 中 (大小为 32)
//
// 然后每个 group 循环 log2(32) 次, 每次由 group 中一半的线程进行一个 reduce:
//
// 1 5 9 13 ...       129 133 137 141 ...
// 6   21   ...       262     278     ...
// 28       ...       540             ...
// .                  .
// .                  .
// 2016               6112
//
// 最后由 group 的第一个 thread (local_linear_id 为 0) 负责把当前 group 的 sum
// 写到 global_memory[group_id] 的位置
//
// 所以第一次循环过后 global memory 会变成:
//
// 2016 6112 2 3 4 5 ... 127
//
// 下一个循环会只启动一个 group, 虽然仍然有 32 个线程, 但只有第一个线程会工作
// (if
// ((2 * global_id) < len)).
//
// 最终 global_memory[0] 是 reduce 的结果
//
// Q: 每个 workgroup 大小如何确定? 为什么这里选择 32?
// A: 总线程数是相同的, 但更大的 workgroup 可以更多的使用 local memory,
// 可以减少访存 及同步开销, 但最终 workgroup 大小受限于 thread 多少及 local
// memory 大小
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability)

#define N 1026
int main(int, char**) {
    std::array<int32_t, N> arr;
    std::cout << "Data: ";
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    sycl::buffer<int32_t, 1> buf(arr.data(), sycl::range<1>(arr.size()));

    sycl::device device = sycl::host_selector{}.select_device();

    sycl::queue queue(device, [](sycl::exception_list el) {
        for (auto ex : el) {
            std::rethrow_exception(ex);
        }
    });

    size_t wgroup_size = 32;

    auto part_size = wgroup_size * 2;

    auto has_local_mem =
        device.is_host() ||
        (device.get_info<sycl::info::device::local_mem_type>() !=
         sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t))) {
        throw "Device doesn't have enough local memory!";
    }

    // <<Reduction loop>>
    auto len = arr.size();
    while (len != 1) {
        // division rounding up
        auto n_wgroups = (len + part_size - 1) / part_size;
        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor<
                int32_t, 1, sycl::access::mode::read_write,
                sycl::access::target::local>
                local_mem(sycl::range<1>(wgroup_size), cgh);

            auto global_mem =
                buf.get_access<sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<class reduction_kernel>(
                sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
                [=](sycl::nd_item<1> item) {
                    size_t local_id = item.get_local_linear_id();
                    size_t global_id = item.get_global_linear_id();
                    local_mem[local_id] = 0;
                    if ((2 * global_id) < len) {
                        local_mem[local_id] = global_mem[2 * global_id];
                    }
                    if ((2 * global_id + 1) < len) {
                        local_mem[local_id] += global_mem[2 * global_id + 1];
                    }
                    // if ((2 * global_id) < len) {
                    //     local_mem[local_id] = global_mem[2 * global_id] +
                    //                           global_mem[2 * global_id + 1];
                    // }
                    // NOTE: barrier 有两个作用
                    // 1. 做为 barrier, 确保当前 work_group 的所有 work_item
                    // 都执行到这个地方
                    // 2. 做为 mem fence, 确保所有对 local buffer
                    // 的修改都已经生效了
                    //
                    // ps. fence_space 可以是 local_space, global_space 或
                    // global_and_local, 是指 mem fence 影响的范围: local buffer
                    // 还 是 global buffer
                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
                        auto idx = 2 * stride * local_id;
                        if (idx < wgroup_size) {
                            local_mem[idx] =
                                local_mem[idx] + local_mem[idx + stride];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_id == 0) {
                        global_mem[item.get_group_linear_id()] = local_mem[0];
                    }
                });
        });
        queue.wait_and_throw();

        len = n_wgroups;
    }

    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}
