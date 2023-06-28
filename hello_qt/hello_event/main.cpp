#include <QCoreApplication>
#include <QTimer>
#include "my_thread.h"
#include "simple_object.h"
#include <QEvent>

// https://wiki.qt.io/Threads_Events_QObjects

int main(int argc, char *argv[]) {
    QCoreApplication a(argc, argv);
    /* 1. myThread 需要在 run 里执行 exec 才能进入 event loop */
    MyThread myThread;
    myThread.start();

    /* 2. thread 默认的实现是 void run() {exec();}, 所以 QThread 默认会进入
          event loop */
    QThread thread;
    thread.start();

    /* 3. moveToThread 可以把 qobject 与 thread 绑定, obj 对应的 slot 会通过 event
     * 的形式在相应的 thread 执行 */
    SimpleObject obj;
    obj.moveToThread(&myThread);

    /* 4. 这里会失败, 因为整个 qobject 树的 object 必须在绑定到同一个 thread, 因
          为涉及到 object 释放的问题 */
    SimpleObject obj2(&obj);
    obj2.moveToThread(&thread);

    /* 5. Timer 是通过直接向 thread event loop 注册 timer 实现的, 并不需要单独的
          线程 */
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, &obj, &SimpleObject::handleEvent);
    timer.start(1000);

    qDebug() << "main run:" << QThread::currentThreadId();

    /* 6. handleLongEvent 会长时间运行以致阻塞主线程的 event loop, 所以
       handleLongEvent 中需要通过 QCoreApplication::processEvents 来强行处理
       event */
    QTimer timer2;
    timer2.setSingleShot(true);
    timer2.callOnTimeout(&obj, &SimpleObject::handleLongEvent);
    timer2.start(2000);

    // 7. slot 的执行在底层是通过 event loop 的 MetaCall event 实现的. postEvent
    // 可以直接发送一个特定的 event, QObject::event() 负责处理 event
    QCoreApplication::postEvent(&obj, new QEvent(QEvent::None));

    // 8. a.exec 底层的代码是 {QEventLoop loop; loop.exec();}
    return a.exec();
}
