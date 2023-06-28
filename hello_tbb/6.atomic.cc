#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>
#include <utility>

using namespace std;
using namespace tbb;

#define N 3
int main(int argc, char *argv[]) {
    cout << "-----" << endl;
    atomic<int> sum(0);
    parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); i++) {
            sum.fetch_add(i);
        }
    });
    cout << sum << endl;
    return 0;
}
