#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/tbb.h>

#include <iostream>

using namespace std;
using namespace tbb;

int fib(int n) {
    if (n < 2) {
        return n;
    } else {
        int x, y;
        task_group g;
        g.run([&] { x = fib(n - 1); });
        g.run([&] { y = fib(n - 2); });
        g.wait();
        return x + y;
    }
}

#define N 5
int main(int argc, char* argv[]) {
    cout << "-----" << endl;
    cout << fib(N) << endl;

    return 0;
}
