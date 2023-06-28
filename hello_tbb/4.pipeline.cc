#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>

using namespace std;
using namespace tbb;

#define N_TOKEN 3
#define N 10

int main(int argc, char* argv[]) {
    cout << "-----" << endl;
    int n = 0;
    auto f1 = make_filter<void, int>(
        tbb::filter_mode::serial_in_order, [&](flow_control& fc) {
            if (n++ == N) {
                fc.stop();
            }
            return n;
        });
    auto f2 = make_filter<int, void>(
        tbb::filter_mode::parallel, [=](int i) { cout << i << endl; });
    parallel_pipeline(N_TOKEN, f1 & f2);

    return 0;
}
