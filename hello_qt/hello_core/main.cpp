#include <QCoreApplication>
#include <QVector>

void detach() {
    QVector<int> data {1, 2, 3};
    // data.setSharable(false);
    printf("%p\n", &data[0]);

    auto dataCopy = data;
    // printf("%p\n", &dataCopy[0]);
    // printf("-------\n");
    // for (auto it = dataCopy.begin(); it != dataCopy.end(); ++it) {
    //     printf("%p\n", it);
    // }
    printf("-------\n");
    for (int &i : dataCopy) {
        printf("%p\n", &i);
    }

}

void change() {
    QVector<int> data{1, 2, 3};
    int &first = data[0];

    auto data2 = data;
    first = 2;

    foreach(int i, data2) {
        printf("%d\n", i);
    }
}

void test_foreach() {
    QVector<int> data{1, 2};
    data.reserve(100);

    for(int &i : data) {
        printf("%p %d\n", &i, i);
    }
    printf("---\n");
    foreach (int i, data) {
        data.insert(0, i);
    }

    for(int &i : data) {
        printf("%p %d\n", &i, i);
    }

    printf("---\n");

    for(int &i : data) {
        data.insert(0, i);
    }

    for(int &i : data) {
        printf("%p %d\n", &i, i);
    }
    printf("---\n");
    for (auto it = data.begin(); it != data.end(); ++it) {
        data << *it;
    }
}

void test_for_i() {
    QVector<int> data{1, 2, 3};
    for(int &i : data) {
        printf("%p %d\n", &i, i);
    }
    printf("---\n");
    auto data2 = data;
    for(const int &i : qAsConst(data2)) {
        printf("%p %d\n", &i, i);
    }
    printf("---\n");
    for(int &i : data2) {
        printf("%p %d\n", &i, i);
    }
    printf("---\n");
    for(int &i : data2) {
        i += 1;
    }
    for(int &i : data2) {
        printf("%p %d\n", &i, i);
    }
}
int main(int argc, char *argv[]) {
    // detach();
    // change();
    // test_foreach();
    test_for_i();
    return 0;
}
