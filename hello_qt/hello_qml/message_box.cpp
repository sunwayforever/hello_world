#include "message_box.h"

#include <QDebug>

MessageBox::MessageBox(QObject *parent) : QObject(parent) {
}

int MessageBox::width() {
    return mWidth;
}

void MessageBox::setWidth(int x) {
    mWidth = x;
    qDebug() << "set width: " << mWidth;
    emit oopsWidthChanged();
}

void MessageBox::onTextChanged(QString text) {
    qDebug() << "onTextChanged from Message_box: " << text;
}

void MessageBox::foo() {
    qDebug() << "invoke foo";
}
