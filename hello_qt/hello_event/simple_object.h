// 2020-01-03 14:45
#ifndef SIMPLE_OBJECT_H
#define SIMPLE_OBJECT_H

#include <QDebug>
#include <QObject>
#include <QCoreApplication>
#include <QThread>
#include <QEvent>

class SimpleObject : public QObject {
    Q_OBJECT

  public:
    SimpleObject(QObject *parent = 0) : QObject(parent) {}

  signals:

  public slots:
    void handleEvent() {
        qDebug() << "handle event";
    }
  public:

    void hello() {
        qDebug() << "hello";
    }
    void handleLongEvent() {
        while (true) {
            qDebug() << "handle long time event";
            QCoreApplication::processEvents();
            QThread::sleep(5);
        }
    }

    bool event(QEvent *event) {
        qDebug() << "got event" << event->type();
        return QObject::event(event);
    }
};


#endif  // SIMPLE_OBJECT_H
