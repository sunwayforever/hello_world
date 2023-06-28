#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>
#include <utility>

using namespace std;
using namespace tbb;

#define N 3
int main(int argc, char *argv[]) {
    cout << "-----" << endl;
    concurrent_vector<int> v;
    parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); i++) {
            v.push_back(i);
        }
    });
    for (auto it = v.begin(); it != v.end(); ++it) {
        cout << *it << endl;
    }
    return 0;
}
