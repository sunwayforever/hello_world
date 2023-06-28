// 2020-01-03 14:36
#ifndef MY_THREAD_H
#define MY_THREAD_H
#include <QDebug>
#include <QThread>

class MyThread : public QThread {
  public:
    void run() {
        qDebug() << "my thread run: " << currentThreadId();
        exec();
    }
};

#endif  // MY_THREAD_H
