#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_vector_add;
class kernel_vector_add_2;
int main(int argc, char* argv[]) {
    // NOTE: 这里是 application scope
    // <<Setup host storage>>
    // float4 是 sycl::vec<float,4> 的别名
    // NOTE: sycl::vec 并不是类似 std::vector 的实现, 因为它只支持 1,2,3,4,8,16
    // 个元素, 且数据只能是基本的 scalar type, 它可以直接映射为 opencl 的 vector
    // 类型例如 cl_float16. sycl::vec 可以作为一个整体进行操作, 例如
    // +,-,*,/,==,=,~,<<, > 等
    sycl::float4 a = {1.0, 2.0, 3.0, 4.0};
    sycl::float4 b = {1.0, 2.0, 3.0, 4.0};
    sycl::float4 c = {0.0, 0.0, 0.0, 0.0};
    // <<Initialize device selector>>
    sycl::gpu_selector device_selector;
    // <<Initialize queue>>
    sycl::queue queue(device_selector);
    {
        // <<Setup device storage>>
        // sycl::range<dims>(dim1,dim2,...)
        // NOTE: buffer 必须定义在 application scope
        sycl::buffer<sycl::float4, 1> buff_a(&a, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> buff_b(&b, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> buff_c(&c, sycl::range<1>(1));
        //   <<Execute kernel>>
        queue.submit([&](sycl::handler& cgh) {
            // NOTE: 这里是 command group scope
            // accessor 必须定义在 command group scope
            auto a_acc = buff_a.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = buff_b.get_access<sycl::access::mode::read>(cgh);
            auto c_acc =
                buff_c.get_access<sycl::access::mode::discard_write>(cgh);

            // NOTE: single_task 表示只启动一个线程
            // 更常用的是 parallel_for 以启动多个线程
            // [=] 表示 lambda 使用 copy 来 capture 自由变量
            // [&] 表示 lambda 使用 refernece 来 capture 自由变量
            // sycl 的 kernel 只支持 [=] 而不支持 [&]
            // single_task 相当于
            // cgh.parallel_for<class kernel_vector_add>(
            //     sycl::nd_range<1>(1, 1),
            //     [=](sycl::nd_item<1> item) { c_acc[0] = a_acc[0] + b_acc[0];
            //     });
            // NOTE: 一个 command group 只能有一个 kernel function
            // NOTE: kernel function 必须有一个名字, 以标识 kernel compiler
            // 生成的数据. 对于 kernel function class 来说直接用 class 名字即可,
            // 对于没名字的 lambda, sycl 要求提供一个模板参数做为名字
            // (比如这里的 class kernel_vector_add)
            // NOTE: kernel function 的参数(或者 lambda 能 capture 的数据)
            // 都是传值的, 不能包含指针, 且需要是 POD 类型
            cgh.single_task<class kernel_vector_add>([=]() {
                // NOTE: 这里是 kernel scope
                c_acc[0] = a_acc[0] + b_acc[0];
            });
        });

        // NOTE: queue.submit 后是异步的, 这里直接访问还没有结果.
        // std::cout << c.x() << "," << c.y() << "," << c.z() << "," << c.w()
        //           << std::endl;
        //
        // 原始的 buffer `c` 需要等待异步执行完毕并且 device copy
        // 回来才能拿到结果
        //
        // 有两种方式可以用来等待异常执行的结果:
        //
        // 1. 通过 c++ 作用域, 当 buffer_c 离开作用域是, sycl 会强制一个
        // `等待结果` 和 `device copy` 的动作
        // 2. 通过 host accessor 表示对数据的依赖
    }

    // <<Print results>>
    std::cout << c.x() << "," << c.y() << "," << c.z() << "," << c.w()
              << std::endl;

    // NOTE: 通过 host accessor 表示数据的依赖及读取数据
    c = {0.0, 0.0, 0.0, 0.0};
    sycl::buffer<sycl::float4, 1> buff_a(&a, sycl::range<1>(1));
    sycl::buffer<sycl::float4, 1> buff_b(&b, sycl::range<1>(1));
    sycl::buffer<sycl::float4, 1> buff_c(&c, sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = buff_a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = buff_b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = buff_c.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class kernel_vector_add_2>(
            // c_acc[0] 是因为 float4 类型只有一个元素 (虽然它包含 4 个数)
            [=]() { c_acc[0] = a_acc[0] + b_acc[0]; });
    });
    // NOTE: c_acc 表示的对 buff_c 的依赖会要求 sycl 在这里有一个 wait,
    // 等等异步操作的结束
    auto c_acc = buff_c.get_access<sycl::access::mode::read>();
    std::cout << c_acc[0].x() << "," << c_acc[0].y() << "," << c_acc[0].z()
              << "," << c_acc[0].w() << std::endl;
    // NOTE: 由于 buff_c 还有效, 所以 `c` 会由 buffer_c 接管, 对 `c`
    // 的直接访问无效
    // std::cout << c.x() << "," << c.y() << "," << c.z() << ","
    // << c.w() << std::endl;
    return 0;
}
