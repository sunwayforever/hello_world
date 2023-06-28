#include <CL/sycl.hpp>
#include <iostream>
namespace sycl = cl::sycl;

class kernel_dummy_1;
class kernel_dummy_2;
class kernel_dummy_3;
class kernel_dummy_4;
class kernel_dummy_5;
void test_buffer_dep() {
    // A=a*2
    // B=b*3
    // C=A+B
    int32_t a = 1;
    int32_t b = 2;
    int32_t A = 0;
    int32_t B = 0;
    int32_t C = 0;

    sycl::queue queue(sycl::gpu_selector{});

    sycl::buffer<int32_t, 1> buff_a(&a, sycl::range<1>(1));
    sycl::buffer<int32_t, 1> buff_b(&b, sycl::range<1>(1));
    sycl::buffer<int32_t, 1> buff_A(&A, sycl::range<1>(1));
    sycl::buffer<int32_t, 1> buff_B(&B, sycl::range<1>(1));
    sycl::buffer<int32_t, 1> buff_C(&C, sycl::range<1>(1));
    // NOTE: 在 cuda 中, 同一个 stream 里的 kernel 都是 in-order 执行的. 但是在
    // sycl 中, 按照 sycl 规范, kernel 总是 out-of-order 执行的, out-of-order 真
    // 正是如何执行的取决于 accessor 决定的数据依赖关系 (DAG)
    //
    // 需要注意的是 sycl runtime 不可能完全自己推导出来 DAG, DAG 和 用户 submit
    // 的顺序有关. 例如, 若 submit 的kernel 顺序变成 (C=A+B, A=2a, B=2B), 则 C
    // 的结果会是错误的 0 (而不是 8)
    //
    // 原因在于, 即使多个 kernel 中有些是 `read A`, 有些是 `write A`, sycl
    // runtime 也并不能推导出 `read A` 依赖 `write A`, 需要用户通过 submit
    // 来告诉 runtime 针对 `A` 的 data race 是 read-write 还是 write-read.
    //
    // 这一点与普通 cpu 的指令重排非常类似:
    //
    // 1. x=a+1
    // 2. y=b+1
    // 3. z=x+y
    //
    // 编译器通过分析代码能知道 (1,3),(2,3) 有 data race (都是
    // read-after-write), 但 (1,2) 并没有, 所以 1,2可以自由的乱序执行
    queue.submit([&](sycl::handler& cgh) {
        auto b_acc = buff_b.get_access<sycl::access::mode::read>(cgh);
        auto B_acc = buff_B.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class kernel_dummy_1>([=]() {
            printf("B=b*3\n");
            B_acc[0] = b_acc[0] * 3;
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = buff_a.get_access<sycl::access::mode::read>(cgh);
        auto A_acc = buff_A.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class kernel_dummy_3>([=]() {
            printf("A=a*2\n");
            A_acc[0] = a_acc[0] * 2;
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto A_acc = buff_A.get_access<sycl::access::mode::read>(cgh);
        auto B_acc = buff_B.get_access<sycl::access::mode::read>(cgh);
        auto C_acc = buff_C.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class kernel_dummy_2>([=]() {
            printf("C=A+B\n");
            C_acc[0] = A_acc[0] + B_acc[0];
        });
    });

    auto host_acc = buff_C.get_access<sycl::access::mode::read>();
    printf("%d\n", host_acc[0]);
}

void test_sub_buffer_dep() {
    int data[10] = {0};
    sycl::buffer<int, 1> buf(data, sycl::range<1>(10));
    sycl::buffer<int, 1> sub_buf_1(buf, sycl::id<1>(0), sycl::range<1>(5));
    sycl::buffer<int, 1> sub_buf_2(buf, sycl::id<1>(5), sycl::range<1>(5));

    sycl::queue queue(sycl::host_selector{});

    // NOTE: 由于 sub_buf_1 与 sub_buf_2 没有重合, 所以下面两个 kernel 执行的顺
    // 序可以是不确定的
    queue.submit([&](sycl::handler& cgh) {
        auto acc = sub_buf_1.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class kernel_dummy_4>(
            sycl::range<1>(5), [=](sycl::item<1> item) {
                printf("sub_buf_1\n");
                acc[item.get_id()] = 1;
            });
    });
    queue.submit([&](sycl::handler& cgh) {
        auto acc = sub_buf_2.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class kernel_dummy_5>(
            sycl::range<1>(5), [=](sycl::item<1> item) {
                printf("sub_buf_2\n");
                acc[item.get_id()] = 2;
            });
    });
    queue.wait();
}

int main(int argc, char* argv[]) {
    test_buffer_dep();
    test_sub_buffer_dep();
    return 0;
}
