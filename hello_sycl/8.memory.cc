#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_dummy;

void test_different_accessor() {
    std::array<float, 10> a = {1.0, 2.0, 3.0, 4.0, 5.0,
                               6.0, 7.0, 8.0, 9.0, 10.0};
    sycl::gpu_selector device_selector;
    sycl::queue queue(device_selector, [](sycl::exception_list el) {
        for (auto ex : el) {
            try {
                std::rethrow_exception(ex);
            } catch (sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    });

    {
        sycl::buffer<float, 1> buff_a(
            a.data(), sycl::range<1>(10),
            {sycl::property::buffer::use_host_ptr()});

        // NOTE: buffer 并不需要一定指定 host data. 当指定了 host data 时, sycl
        // runtime 会负责用 host data 初始化所有的 accessor 的数据, 并且把数据写
        // 回到 host ptr. buffer 对象本身并没有对应的 storage, storage 是与
        // accessor 绑定在一起, buffer 只是用来同步 accessor 的数据
        sycl::buffer<float, 1> buff_b(sycl::range<1>(10));

        queue.submit([&](sycl::handler& cgh) {
            // accessor 有 4 种 target:
            //
            // 1. global_buffer
            // 2. host_buffer
            // 3. consant_buffer
            // 4. local
            //
            // NOTE: 指定 cgh 参数时, target 默认是 global_buffer
            //
            // auto a_acc = buff_a.get_access<
            //     sycl::access::mode::read,
            //     sycl::access::target::global_buffer>( cgh);
            //
            // 除了 buffer.get_access, 还可以通过 accessor 构造函数:
            //
            // sycl::accessor<
            //     float, 1, sycl::access::mode::read,
            //     sycl::access::target::global_buffer>
            //     a_acc(buff_a, cgh);
            //
            // NOTE: 不指定 handle 时默认使用 host_buffer
            //
            // auto a_acc = buff_a.get_access<
            //     sycl::access::mode::read>();
            //
            // sycl::accessor<
            //     float, 1, sycl::access::mode::read,
            //     sycl::access::target::host_buffer>
            //     a_acc(buff_a);
            //
            // NOTE: constant_buffer 只支持 sycl::access::mode::read, 不支持
            // write: sycl::accessor<
            //     float, 1, sycl::access::mode::read,
            //     sycl::access::target::constant_buffer>
            //     const_acc(buff_a, cgh);
            //
            // NOTE: local_buffer 是 workgroup 内部 buffer, sycl 不支持通过初始
            // 数据来初始化, 只能指定一个大小. host 无法访问 local_buffer
            //
            // sycl::accessor<
            //     float, 1, sycl::access::mode::read_write,
            //     sycl::access::target::local>
            //     const_acc(sycl::range<1>(10), cgh);
            //
            // NOTE: private memory 没有对应的 target, kernel
            // 中的局部变量等会自动使用 private memory, 同时 host 也无法访问
            // private memory
            //
            // NOTE: 看起来 USM (malloc_{device,host,shared}) 没有 api 可以从
            // constant_buffer 或 local 分配内存
            //
            // NOTE: command group 中声明的各种 accessor 实际上是声明了 kernel对
            // 数据的依赖关系, sycl runtime 会根据 accessor 来 enqueue 一些涉及
            // 数据拷贝的命令

            sycl::accessor<
                float, 1, sycl::access::mode::read_write,
                sycl::access::target::global_buffer>
                global_acc(buff_a, cgh);

            sycl::accessor<
                float, 1, sycl::access::mode::read_write,
                sycl::access::target::global_buffer>
                global_acc_b(buff_b, cgh);

            sycl::accessor<
                float, 1, sycl::access::mode::read,
                sycl::access::target::constant_buffer>
                const_acc(buff_a, cgh);

            sycl::accessor<
                float, 1, sycl::access::mode::read_write,
                sycl::access::target::local>
                local_acc(sycl::range<1>(5), cgh);

            cgh.parallel_for<class kernel_dummy>(
                sycl::nd_range<1>(10, 5), [=](sycl::nd_item<1> item) {
                    // NOTE: malloc 或 malloc_device 均无法在 kernel 中使用
                    // float* x = (float*)malloc(4);
                    // sycl::malloc_device<float>(1, queue);
                    size_t local_id = item.get_local_linear_id();
                    size_t global_id = item.get_global_linear_id();
                    // 数组可以使用, 但依赖于动态内存分配的 stl 容器无法使用
                    std::array<float, 1> private_buf1 = {const_acc[global_id]};
                    float private_buf2[1] = {const_acc[global_id]};
                    global_acc[global_id] += private_buf1[0] + private_buf2[0];
                    global_acc_b[global_id] +=
                        private_buf1[0] + private_buf2[0];
                    // NOTE: accessor 还有一个 get_pointer 方法, 所以与底层
                    // buffer的绑 定 (global, local, constant, host) 是通过
                    // sycl::accessor, 而 不是 sycl::buffer
                    auto ptr = global_acc.get_pointer();
                    auto const_ptr = const_acc.get_pointer();
                    auto local_ptr = local_acc.get_pointer();
                    *(ptr + global_id) += *(ptr + global_id);

                    printf(
                        "global ptr: %p, constant ptr: %p, local ptr: %p, "
                        "group: "
                        "%d\n",
                        ptr + global_id, const_ptr + global_id,
                        local_ptr + local_id, item.get_group_linear_id());
                });
        });

        sycl::accessor<
            float, 1, sycl::access::mode::read,
            sycl::access::target::host_buffer>
            host_acc(buff_a);

        sycl::accessor<
            float, 1, sycl::access::mode::read,
            sycl::access::target::host_buffer>
            host_acc_b(buff_b);

        auto host_ptr = host_acc.get_pointer();
        // NOTE: host ptr 与 host buffer 不同, 因为默认情况下 accessor
        // 是分配新的内 存 (host 或 device 上), 同一个 sycl::buffer 对应的
        // accessor 会按需要自动完成数据的复制
        printf("host ptr: %p, host buffer: %p", host_ptr, a.data());

        printf("------\n");
        for (int i = 0; i < 10; i++) {
            printf("%f\n", *(host_ptr + i));
        }

        printf("------\n");
        for (int i = 0; i < 10; i++) {
            printf("%f\n", host_acc[i]);
        }

        printf("------\n");
        for (int i = 0; i < 10; i++) {
            printf("%f\n", host_acc_b[i]);
        }

        printf("------\n");
        for (int i = 0; i < 10; i++) {
            printf("%f\n", a[i]);
        }
    }
    // NOTE: buff_a 析构, sycl 会把结果复制到 a
    printf("------\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n", a[i]);
    }
}

void test_smart_pointer() {
    // NOTE: buffer 正常情况下会写回数据到 host_data, 为了避免写回, 可以在初始化
    // buffer 时不指定 host_data, 但这样又无法直接给 accessor 初值 (可以用
    // queue.memcpy 或 cgh.memcpy 来赋值). 如果既想用 host_data 给 accessor
    // 初始化, 又想避免 buffer 写回数据, 可以用:
    // 1. unique_ptr
    // 2. const data
    // 当然使用 unique_ptr 控制写回数据只是它的副作用, unique_ptr
    // 最大的用处还是它作为 smart pointer 用来自动回收资源
    sycl::queue queue(sycl::gpu_selector{});

    int* a = new int[10];
    std::unique_ptr<int, std::default_delete<int[]>> data(a);

    int b[10] = {0};
    int c[10] = {0};
    int d[10] = {0};

    {
        sycl::buffer<int, 1> buf_a(std::move(data), sycl::range<1>(10));
        sycl::buffer<int, 1> buf_b(b, sycl::range<1>(10));
        sycl::buffer<int, 1> buf_c(b, sycl::range<1>(10));
        // NOTE: buffer.set_write_back 也可以阻止 data 的写回
        buf_b.set_write_back(false);
        // NOTE: buffer.set_final_data 可以控制写回到什么地方
        buf_c.set_final_data(c);

        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = buf_a.get_access<sycl::access::mode::read_write>(cgh);
            auto b_acc = buf_b.get_access<sycl::access::mode::read_write>(cgh);
            auto c_acc = buf_c.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class kernel_dummy_2>(
                sycl::range<1>(10), [=](sycl::item<1> item) {
                    a_acc[item.get_id()] += 1;
                    b_acc[item.get_id()] = 1;
                    c_acc[item.get_id()] = 1;
                });
            // NOTE: cgh 还提供了 copy, fill 可以直接读写 accessor 和 host ptr
            cgh.copy(c_acc, d);
        });
        // NOTE: buf_a 析构时无法写回数据到 data, 因为 application scope
        // 已经无法访问 data
    }
    for (int i = 0; i < 10; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", d[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    test_different_accessor();
    printf("---------------\n");
    test_smart_pointer();
    return 0;
}
