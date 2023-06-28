#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "backend.h"
#include "complex_message_box.h"
#include <QDebug>

int main(int argc, char *argv[]) {
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    Backend backend;
    QGuiApplication app(argc, argv);

    qmlRegisterType<ComplexMessageBox>("sunway.message_box", 1, 0, "MessageBox");

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("backend", &backend);
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    engine.load(url);

    QObject* window = engine.rootObjects().first();
    QObject::connect(window, SIGNAL(windowChanged()), &backend, SLOT(onWindowChanged()));

    emit backend.interrupted();

    return app.exec();
}
