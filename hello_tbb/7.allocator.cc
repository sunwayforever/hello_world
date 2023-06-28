#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/scalable_allocator.h>
#include <tbb/tbb.h>

#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using namespace tbb;

#define N 3
int main(int argc, char *argv[]) {
    cout << "-----" << endl;
    // NOTE: cache_aligned_allocator 用来避免 false sharing
    vector<int, cache_aligned_allocator<int> > v;
    // NOTE: scaleable_allocator 是一个多线程的 malloc
    // vector<int, scalable_allocator<int> > v;
    for (int i = 0; i < N; i++) {
        v.push_back(i);
    }
    parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); i++) {
            v[i] += 1;
        }
    });
    for (auto it = v.begin(); it != v.end(); ++it) {
        cout << *it << endl;
    }
    return 0;
}
