#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>

using namespace std;
using namespace tbb;

class Cout {
   public:
    void operator()(const blocked_range<int> &r) const {
        for (int i = r.begin(); i != r.end(); ++i) {
            cout << i << endl;
        };
    }
};

#define N 3
int main(int argc, char *argv[]) {
    cout << "-----" << endl;
    parallel_for(blocked_range<int>(0, N), Cout());

    cout << "-----" << endl;
    parallel_for(blocked_range<int>(0, N), [](const blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); i++) {
            cout << i << endl;
        }
    });

    cout << "-----" << endl;
    parallel_for(0, N, [](int i) { cout << i << endl; });

    return 0;
}
