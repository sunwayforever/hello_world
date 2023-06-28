#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>

using namespace std;
using namespace tbb;

#define N 3
int main(int argc, char *argv[]) {
    cout << "-----" << endl;
    // NOTE: simple_partitioner 类似于 omp 的 static scheduler, G 相当于
    // schedule(static, STEP) 中的 step
    int G = 1;
    parallel_for(
        blocked_range<int>(0, N, G),
        [=](const blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); i++) {
                cout << i << endl;
            }
        },
        simple_partitioner());

    cout << "-----" << endl;
    // NOTE: affinity_partitioner 能更好的利用 cache. 例如两次 parallel_for 访问
    // 相同的数据时, 用 ap 可以让同一个线程两次都分到相同的数据
    affinity_partitioner ap;
    parallel_for(
        0, N, [=](int i) { cout << i << endl; }, ap);
    parallel_for(
        0, N, [=](int i) { cout << i << endl; }, ap);

    cout << "-----" << endl;
    parallel_for(
        0, N, [=](int i) { cout << i << endl; }, auto_partitioner());
    return 0;
}
