#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>

using namespace std;
using namespace tbb;

class Sum {
   public:
    int sum;
    void operator()(const blocked_range<int>& r) {
        // NOTE: operator() 被调用时 sum 可能已经通过 join 获得到了值, 因为多个
        // Sum 可能是线性执行的, 例如: sum1 -> join -> sum2 -> join -> sum3, 而
        // 不是
        //    sum1 --+
        //    sum2 --+--> join
        //    sum2 --+
        for (int i = r.begin(); i != r.end(); ++i) {
            sum += i;
        }
    }
    void join(const Sum& y) { sum += y.sum; }
    Sum(Sum& x, split dummy) : sum(0) {}
    Sum() : sum(0) {}
};

#define N 3
int main(int argc, char* argv[]) {
    cout << "-----" << endl;
    int sum = parallel_reduce(
        blocked_range<int>(0, N), 0,
        // NOTE: partial_sum 代表 sum 前已经通过 join 获得的中间结果
        [=](const blocked_range<int>& r, int partial_sum) {
            for (int i = r.begin(); i != r.end(); i++) {
                partial_sum += i;
            }
            return partial_sum;
        },
        std::plus<int>());
    cout << sum << endl;

    cout << "-----" << endl;
    Sum x;
    parallel_reduce(blocked_range<int>(0, N), x);
    cout << x.sum << endl;
    return 0;
}
