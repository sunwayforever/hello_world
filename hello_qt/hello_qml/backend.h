#ifndef BACKEND_H
#define BACKEND_H


#include <QObject>

class Backend : public QObject {
    Q_OBJECT

  signals:
    void interrupted();

  public:
    explicit Backend(QObject *parent = 0);

  public slots:
    void onTextChanged(QString);
    void onWindowChanged();
};

#endif  // BACKEND_H
