#include "backend.h"

#include <QDebug>

Backend::Backend(QObject *parent) : QObject(parent) {
}

void Backend::onTextChanged(QString text) {
    qDebug() << "onTextChanged " << text;
}

void Backend::onWindowChanged() {
    qDebug() << "onWindowChanged";
}
